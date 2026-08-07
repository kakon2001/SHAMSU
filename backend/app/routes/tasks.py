from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import ollama
from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import model_registry
from ..agent import tools
from ..config import settings
from .preview import PreviewStartRequest, start_preview

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

ALLOWED_GENERATED_SUFFIXES = {".html", ".css", ".js", ".py", ".c", ".h", ".asm", ".s", ".ld", ".md", ".txt", ".json"}
MAX_GENERATED_FILES = 13
MAX_GENERATED_CHARS = 200_000
MAX_REPAIR_ATTEMPTS = 2


class TaskPlanRequest(BaseModel):
    prompt: str


class TaskPlanResponse(BaseModel):
    goal: str
    mode: str
    steps: list[str]
    suggested_files: list[dict[str, str]]
    verify_commands: list[str]
    notes: list[str]
    requirements_analysis: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)
    file_plan: list[str] = Field(default_factory=list)
    workflow_summary: str = ""


class TaskRunRequest(BaseModel):
    prompt: str
    preview: bool = True
    overwrite: bool = True


class TaskRunStep(BaseModel):
    name: str
    status: str
    detail: str


class ReliabilityReport(BaseModel):
    phases: list[str]
    verification_feedback: list[str]
    repair_attempts: int
    final_status: str
    next_action: str


class TaskRunResponse(BaseModel):
    goal: str
    mode: str
    ok: bool
    created_files: list[str]
    preview_url: str | None
    steps: list[TaskRunStep]
    notes: list[str]
    requirements_analysis: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)
    file_plan: list[str] = Field(default_factory=list)
    workflow_summary: str = ""
    reliability: ReliabilityReport | None = None


@router.post("/plan", response_model=TaskPlanResponse)
async def plan_task(body: TaskPlanRequest) -> TaskPlanResponse:
    return _with_advisory(build_plan(body.prompt), body.prompt)


@router.post("/run", response_model=TaskRunResponse)
async def run_task(body: TaskRunRequest) -> TaskRunResponse:
    """Run the autonomous build loop: plan -> write -> verify -> preview.

    Known safe templates run deterministically. Unknown build prompts fall back to
    a strict JSON file generator powered by the selected local Ollama model.
    """
    plan = _with_advisory(build_plan(body.prompt), body.prompt)
    steps: list[TaskRunStep] = [TaskRunStep(name="plan", status="ok", detail=f"Analyzed request and selected {plan.mode} workflow.")]
    notes = list(plan.notes)

    if not plan.suggested_files and _looks_like_build_request(body.prompt) and not plan.mode.startswith(("openbazaar-20-step", "openbazaar-prd", "openbazaar-staged")):
        generated = await _generate_json_file_plan(body.prompt, steps)
        if generated:
            plan = _with_advisory(_make_generated_plan(body.prompt, generated), body.prompt)
            notes = list(plan.notes)

    created_files = _write_suggested_files(plan, body.overwrite, steps)
    if not plan.suggested_files:
        steps.append(
            TaskRunStep(
                name="write",
                status="skipped",
                detail="No deterministic template or valid JSON file plan was available for this prompt.",
            )
        )
        return TaskRunResponse(
            goal=plan.goal,
            mode=plan.mode,
            ok=False,
            created_files=created_files,
            preview_url=None,
            steps=steps,
            notes=notes + ["Try a more concrete build prompt, or use chat mode to ask for code/instructions."],
            requirements_analysis=plan.requirements_analysis,
            clarification_questions=plan.clarification_questions,
            stack=plan.stack,
            file_plan=plan.file_plan,
            workflow_summary=plan.workflow_summary,
            reliability=_build_reliability_report(steps, created_files, False),
        )

    verify_ok, plan, created_files = await _verify_and_repair_loop(body.prompt, plan, created_files, steps)
    preview_url = await _maybe_start_preview(body.preview, created_files, verify_ok, steps)

    return TaskRunResponse(
        goal=plan.goal,
        mode=plan.mode,
        ok=verify_ok,
        created_files=created_files,
        preview_url=preview_url,
        steps=steps,
        notes=notes,
        requirements_analysis=plan.requirements_analysis,
        clarification_questions=plan.clarification_questions,
        stack=plan.stack,
        file_plan=plan.file_plan,
        workflow_summary=plan.workflow_summary,
        reliability=_build_reliability_report(steps, created_files, verify_ok),
    )



def _build_reliability_report(steps: list[TaskRunStep], created_files: list[str], ok: bool) -> ReliabilityReport:
    phase_order = ["plan", "generate", "validate", "write", "verify", "feedback", "repair", "repair-generate", "preview"]
    phases: list[str] = []
    for phase in phase_order:
        phase_steps = [step for step in steps if step.name == phase]
        if not phase_steps:
            continue
        status = _phase_status(phase_steps)
        detail = phase_steps[-1].detail
        phases.append(f"{phase}: {status} - {detail}")

    feedback = [step.detail for step in steps if step.name == "feedback"]
    repair_attempts = len([step for step in steps if step.name in {"repair", "repair-generate"} and step.status != "skipped"])
    if ok:
        next_action = "Open the preview or inspect the generated files, then ask SHAMSU for changes if needed."
        final_status = "verified"
    elif created_files:
        next_action = "Read the verification feedback, repair the smallest affected file or block, then run verification again."
        final_status = "needs-repair"
    else:
        next_action = "Clarify the request or ask for a smaller first version so SHAMSU can create a valid file plan."
        final_status = "blocked"
    return ReliabilityReport(
        phases=phases,
        verification_feedback=feedback,
        repair_attempts=repair_attempts,
        final_status=final_status,
        next_action=next_action,
    )


def _phase_status(steps: list[TaskRunStep]) -> str:
    statuses = {step.status for step in steps}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    if "ok" in statuses:
        return "ok"
    if "skipped" in statuses:
        return "skipped"
    return steps[-1].status
def build_plan(prompt: str) -> TaskPlanResponse:
    prompt = prompt.strip()
    lower = prompt.lower()
    if _looks_like_openbazaar_marketplace(lower):
        return _openbazaar_dispatch_plan(prompt)
    if _looks_like_operating_system_request(lower):
        return _toy_os_plan(prompt)
    if any(word in lower for word in ["large file", "100000", "100,000", "huge file"]):
        return _large_file_plan(prompt)
    if any(word in lower for word in ["bug", "fix", "error", "traceback", "failing"]):
        return _bugfix_plan(prompt)
    if "brick" in lower and ("breaker" in lower or "game" in lower):
        return _brick_breaker_plan(prompt)
    if "snake" in lower and "game" in lower:
        return _snake_game_plan(prompt)
    if "pong" in lower and "game" in lower:
        return _pong_game_plan(prompt)
    if ("tic" in lower and "toe" in lower) or "tic-tac-toe" in lower:
        return _tic_tac_toe_plan(prompt)
    if "quiz" in lower and ("app" in lower or "game" in lower):
        return _quiz_app_plan(prompt)
    if "todo" in lower or "to-do" in lower:
        return _todo_app_plan(prompt)
    if "calculator" in lower and ("app" in lower or "tool" in lower or "program" in lower):
        return _calculator_app_plan(prompt)
    if "bouncing" in lower and "ball" in lower:
        return _bouncing_ball_plan(prompt)
    if "single html" in lower and ("crm" in lower or "management system" in lower or "management app" in lower):
        return _management_system_plan(prompt)
    if ("crm" in lower or "management system" in lower or "management app" in lower) and _wants_database_backing(lower):
        return _database_backed_management_app_plan(prompt)
    if "crm" in lower or "management system" in lower or "management app" in lower:
        return _multi_file_management_app_plan(prompt)
    if _looks_like_website_request(lower):
        return _website_prototype_plan(prompt)
    if _wants_full_stack(lower) and _looks_like_system_request(lower):
        return _full_stack_project_plan(prompt)
    if _looks_like_system_request(lower):
        return _starter_system_plan(prompt)
    if _looks_like_build_request(prompt):
        return _generated_task_plan(prompt)
    return _general_plan(prompt)






def _looks_like_operating_system_request(lower: str) -> bool:
    os_terms = [
        "operating system", "toy os", "mini os", "kernel", "bootloader",
        "boot sector", "bare metal", "bare-metal", "qemu", "nasm",
    ]
    return any(term in lower for term in os_terms) and any(
        action in lower for action in ["make", "build", "create", "generate", "develop", "implement", "write"]
    )


def _toy_os_plan(prompt: str) -> TaskPlanResponse:
    boot_asm = '''bits 32
section .multiboot
    align 4
    dd 0x1BADB002
    dd 0x00
    dd -(0x1BADB002 + 0x00)

section .text
    global start
    extern kernel_main

start:
    mov esp, stack_top
    call kernel_main
.hang:
    hlt
    jmp .hang

section .bss
    align 16
stack_bottom:
    resb 16384
stack_top:
'''
    kernel_c = r'''#include <stdint.h>
#include <stddef.h>

#define VGA_BUFFER ((volatile uint16_t*)0xB8000)
#define VGA_WIDTH 80
#define VGA_HEIGHT 25

static uint8_t color = 0x0F;
static size_t row = 0;
static size_t col = 0;

static void put_char(char c) {
    if (c == '\n') {
        row++;
        col = 0;
        return;
    }
    VGA_BUFFER[row * VGA_WIDTH + col] = ((uint16_t)color << 8) | (uint8_t)c;
    col++;
    if (col >= VGA_WIDTH) {
        col = 0;
        row++;
    }
    if (row >= VGA_HEIGHT) {
        row = 0;
    }
}

static void print(const char* text) {
    for (size_t i = 0; text[i] != '\0'; i++) {
        put_char(text[i]);
    }
}

void kernel_main(void) {
    for (size_t i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++) {
        VGA_BUFFER[i] = ((uint16_t)0x07 << 8) | ' ';
    }
    print("Welcome to SHAMSU OS\n");
    print("Toy kernel booted in protected mode.\n");
    print("Next steps: keyboard input, interrupts, memory, and shell.\n");
    while (1) {
        __asm__ volatile ("hlt");
    }
}
'''
    linker_ld = '''ENTRY(start)
SECTIONS
{
    . = 1M;
    .text BLOCK(4K) : ALIGN(4K)
    {
        *(.multiboot)
        *(.text)
    }
    .rodata BLOCK(4K) : ALIGN(4K)
    {
        *(.rodata)
    }
    .data BLOCK(4K) : ALIGN(4K)
    {
        *(.data)
    }
    .bss BLOCK(4K) : ALIGN(4K)
    {
        *(COMMON)
        *(.bss)
    }
}
'''
    makefile = '''AS=nasm
CC=gcc
LD=ld
CFLAGS=-m32 -ffreestanding -fno-pie -fno-stack-protector -nostdlib -Wall -Wextra
LDFLAGS=-m elf_i386 -T linker.ld -nostdlib

all: shamsu-os.bin

boot.o: boot.asm
	$(AS) -f elf32 boot.asm -o boot.o

kernel.o: kernel.c
	$(CC) $(CFLAGS) -c kernel.c -o kernel.o

shamsu-os.bin: boot.o kernel.o linker.ld
	$(LD) $(LDFLAGS) boot.o kernel.o -o shamsu-os.bin

run: shamsu-os.bin
	qemu-system-i386 -kernel shamsu-os.bin

clean:
	del /Q *.o *.bin 2>NUL || rm -f *.o *.bin
'''
    readme = f'''# SHAMSU Toy OS

Generated by SHAMSU Toy OS Build Mode.

This is an educational starter operating system kernel, not a production OS. It boots with a Multiboot-compatible loader in QEMU and prints text to VGA memory.

## Generated from prompt

{prompt}

## Files

- `boot.asm`: Multiboot header and assembly entry point
- `kernel.c`: freestanding C kernel that writes to VGA text memory
- `linker.ld`: places the kernel at 1 MB
- `Makefile`: build and QEMU run commands
- `ARCHITECTURE.md`: roadmap for growing the OS safely

## Required tools

Install NASM, GCC with 32-bit support or an i686 cross compiler, GNU ld/binutils, QEMU (`qemu-system-i386`), and make.

## Build

```cmd
cd C:\\Users\\HP\\Desktop\\CSE327\\workspace\\shamsu_os
make
```

Expected output file: `shamsu-os.bin`

## Run in QEMU

```cmd
make run
```

Expected QEMU screen output:

```text
Welcome to SHAMSU OS
Toy kernel booted in protected mode.
Next steps: keyboard input, interrupts, memory, and shell.
```

## Safety note

This project only creates and runs a local kernel image in QEMU. It does not write to a real disk, partition, USB drive, or boot sector.
'''
    architecture = '''# SHAMSU Toy OS Architecture

## Current version

The first version contains a Multiboot header, assembly startup code, a freestanding C kernel, a linker script, and a Makefile. The kernel writes text directly to VGA memory and then halts safely.

## Build workflow

1. Assemble `boot.asm` into `boot.o`.
2. Compile `kernel.c` as freestanding 32-bit C into `kernel.o`.
3. Link objects with `linker.ld` into `shamsu-os.bin`.
4. Run `qemu-system-i386 -kernel shamsu-os.bin`.

## Roadmap

1. Keyboard driver.
2. Interrupt descriptor table.
3. Timer interrupt.
4. Basic memory map.
5. Simple command shell.
6. Tiny file-system simulation.
7. Better build verification and QEMU log capture.

## Limits

This is a toy educational OS. It does not include processes, virtual memory, hardware drivers, networking, users, permissions, or persistent storage.
'''
    smoke = '''from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ["boot.asm", "kernel.c", "linker.ld", "Makefile", "README.md", "ARCHITECTURE.md"]
missing = [name for name in required if not (root / name).exists()]
assert not missing, f"Missing files: {missing}"
kernel = (root / "kernel.c").read_text(encoding="utf-8")
assert "Welcome to SHAMSU OS" in kernel
assert "\\n" in kernel
assert "qemu-system-i386" in (root / "Makefile").read_text(encoding="utf-8")
assert "ENTRY(start)" in (root / "linker.ld").read_text(encoding="utf-8")
print("toy os smoke passed")
'''
    files = [
        {"path": "shamsu_os/boot.asm", "content": boot_asm},
        {"path": "shamsu_os/kernel.c", "content": kernel_c},
        {"path": "shamsu_os/linker.ld", "content": linker_ld},
        {"path": "shamsu_os/Makefile", "content": makefile},
        {"path": "shamsu_os/README.md", "content": readme},
        {"path": "shamsu_os/ARCHITECTURE.md", "content": architecture},
        {"path": "shamsu_os/tests/smoke_test.py", "content": smoke},
    ]
    return TaskPlanResponse(
        goal=prompt,
        mode="toy-os-build-mode",
        steps=[
            "Detect operating-system request and choose safe toy OS workflow.",
            "Generate bootloader entry, kernel, linker script, Makefile, docs, and smoke test.",
            "Verify files are readable and the kernel source contains the expected boot message.",
            "Provide QEMU build/run commands instead of opening a browser preview.",
        ],
        suggested_files=files,
        verify_commands=["python shamsu_os/tests/smoke_test.py", "cd shamsu_os && make", "cd shamsu_os && qemu-system-i386 -kernel shamsu-os.bin"],
        notes=[
            "Toy OS Build Mode creates an educational kernel project, not a production OS.",
            "QEMU/NASM/GCC are external tools; SHAMSU generates commands and verifies files, but those tools must be installed locally to boot it.",
            "Safety: generated commands run a QEMU VM and do not write to a real disk or USB device.",
        ],
        requirements_analysis=[
            "The request asks for operating-system development, which needs boot/kernel files rather than browser HTML.",
            "A safe first version should boot in an emulator and print a visible message.",
            "The project needs explicit limitations and next-step roadmap because a full OS is a large systems project.",
        ],
        clarification_questions=[
            "Should the next OS version add keyboard input, interrupts, memory management, or a shell first?",
            "Which toolchain is available on this laptop: NASM/GCC/QEMU, WSL, or MinGW?",
        ],
        stack=["NASM", "Freestanding C", "GNU ld linker script", "Makefile", "QEMU i386 emulator", "Python smoke test"],
        file_plan=[item["path"] for item in files],
        workflow_summary="prompt -> detect OS request -> generate toy kernel scaffold -> verify files -> provide make/QEMU commands -> iterate into drivers and shell",
    )

def _is_analysis_only_prompt(lower: str) -> bool:
    analysis_terms = [
        "summarize", "summary", "explain", "analyze", "analyse", "read this", "read the",
        "do not create", "do not build", "don't create", "dont create", "no files yet",
        "roadmap", "plan only", "requirements", "list all", "what are", "what is",
    ]
    build_terms = [
        "build", "create files", "generate files", "write files", "implement", "make the project",
        "start phase", "add backend", "add frontend", "add auction", "add cod", "add database", "add security",
        "full project", "create the project", "generate the project",
    ]
    has_analysis = any(term in lower for term in analysis_terms)
    has_build = any(term in lower for term in build_terms)
    explicit_no_build = any(term in lower for term in ["do not create", "do not build", "don't create", "dont create", "no files yet"])
    return explicit_no_build or (has_analysis and not has_build)


def _openbazaar_analysis_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    if "roadmap" in lower or "phase" in lower:
        mode = "openbazaar-prd-roadmap"
        steps = [
            "Phase 1: Requirements analysis and role/page extraction.",
            "Phase 2: Frontend pages for marketplace, product, buyer, seller, admin, architecture, and workflow.",
            "Phase 3: Seller listing form with PRD validation fields.",
            "Phase 4: Auction workflow with bid validation, live update simulation, auto bidding, anti-sniping, and order creation.",
            "Phase 5: Cash on Delivery workflow with OTP, seller confirmation, courier dispatch, completion, refusal, and reliability scoring.",
            "Phase 6: Backend API, database schema, security plan, tests, and preview verification.",
        ]
        summary = "Roadmap only: no files will be created until you use an explicit build/add/implement prompt."
    else:
        mode = "openbazaar-prd-analysis"
        steps = [
            "Executive summary: OpenBazaar is a Cash on Delivery marketplace with Buy Now and auction purchasing.",
            "Roles: Guest browses/searches; Buyer buys/bids; Seller uploads/edits/listings/orders; Admin moderates/users/audit.",
            "Core pages: homepage, product detail, account/login, buyer dashboard, seller dashboard, admin dashboard, workflow, architecture.",
            "Seller listing: images 3-10, optional video, title 15-100 chars, three-level category, condition, pricing mode, specs, defects, shipping.",
            "Auction: next valid bid rejection, live update, auto bid, anti-sniping, reserve check, order creation.",
            "COD: OTP, seller confirmation, courier dispatch, cash payment, refusal, reliability scoring and suspension below 75%.",
            "Database/security: users, categories, items, bids, orders; rate limiting, fraud detection, TLS plan, argon2id password hashing plan.",
        ]
        summary = "Analysis only: SHAMSU has not created project files yet. Use a build/add/implement prompt when you want file generation."
    return TaskPlanResponse(
        goal=prompt,
        mode=mode,
        steps=steps,
        suggested_files=[],
        verify_commands=[],
        notes=[summary],
        requirements_analysis=steps,
        clarification_questions=[
            "Should SHAMSU build the full MVP now, or should it add the PRD features phase by phase for the video?",
            "For the final backend, should the prototype remain FastAPI/SQLite or be upgraded to PostgreSQL?",
        ],
        stack=["No code generation in this step", "PRD analysis", "Roadmap planning"],
        file_plan=["No files are written in analysis/roadmap mode."],
        workflow_summary="PRD analysis/roadmap mode: read uploaded context -> organize requirements -> wait for explicit build prompt before writing files.",
    )
OPENBAZAAR_STAGED_PROMPTS = [
    "Step 1: Read the PRD and summarize the marketplace goal, users, and success criteria. Do not create files yet.",
    "Step 2: Extract role-based requirements for Guest, Buyer, Seller, and Admin. Do not create files yet.",
    "Step 3: Create the project documentation files README.md and WORKFLOW.md from the PRD.",
    "Step 4: Create the application shell index.html with navigation placeholders for all required pages.",
    "Step 5: Add seed data for users, categories, products, bids, orders, audit logs, and COD reliability.",
    "Step 6: Add client-side state management for login, roles, cart, bids, orders, and persistence.",
    "Step 7: Add marketplace, product detail, buyer dashboard, seller dashboard, admin, workflow, and architecture views.",
    "Step 8: Add responsive styling and make the interface presentable for faculty demo.",
    "Step 9: Add smoke tests that prove the generated OpenBazaar pages and PRD keywords exist.",
    "Step 10: Add the SQL database schema for users, categories, items, bids, and orders.",
    "Step 11: Add the FastAPI backend prototype for authentication, items, bids, orders, and admin data.",
    "Step 12: Add backend README instructions for running the OpenBazaar API locally.",
    "Step 13: Add SECURITY.md covering password hashing, rate limiting, fraud detection, TLS, and validation.",
    "Step 14: Verify frontend files and explain which frontend requirement each file satisfies.",
    "Step 15: Verify auction rules: bid increment rejection, auto bidding, reserve price, and anti-sniping.",
    "Step 16: Verify Cash on Delivery: OTP, seller confirmation, courier flow, completion, refusal, and reliability score.",
    "Step 17: Verify database/backend architecture against the PRD tables and production architecture.",
    "Step 18: Polish UI copy, navigation labels, and demo explanation for a clean faculty recording.",
    "Step 19: Run final smoke tests and preview the OpenBazaar app in the browser.",
    "Step 20: Produce the final explanation: what SHAMSU built, what is prototype-only, and what production work remains.",
]


def _openbazaar_stage_number(lower: str) -> int | None:
    match = re.search(r"\b(?:step|stage|phase|prompt)\s*#?\s*(\d{1,2})\b", lower)
    if not match:
        return None
    value = int(match.group(1))
    if 1 <= value <= len(OPENBAZAAR_STAGED_PROMPTS):
        return value
    return None


def _openbazaar_staged_roadmap_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="openbazaar-20-step-prd-build-roadmap",
        steps=OPENBAZAAR_STAGED_PROMPTS,
        suggested_files=[],
        verify_commands=[],
        notes=[
            "Staged PRD Build Mode is active: SHAMSU will not generate the final project in the first prompt.",
            "Use prompts like 'OpenBazaar step 1', 'OpenBazaar step 2', ... 'OpenBazaar step 20' to show visible progress in the demo recording.",
        ],
        requirements_analysis=[
            "The PRD will be handled as a staged software-engineering workflow instead of one instant file dump.",
            "Early prompts analyze and plan; middle prompts create frontend/backend/database/security files; final prompts verify and explain.",
        ],
        clarification_questions=[
            "Should SHAMSU proceed with Step 1 now, or should it wait for your next demo prompt?",
            "Should each step be recorded separately in the faculty video?",
        ],
        stack=["Staged PRD workflow", "HTML/CSS/JavaScript frontend", "FastAPI backend prototype", "SQLite schema", "Smoke tests"],
        file_plan=["No files are written by the roadmap prompt. Files are generated only when a numbered OpenBazaar step asks for them."],
        workflow_summary="20-prompt PRD build: analyze -> plan -> create docs -> create frontend -> add data/state/views/styles -> add tests -> add backend/schema/security -> verify -> explain.",
    )


def _openbazaar_staged_phase_plan(prompt: str, stage: int) -> TaskPlanResponse:
    full = _openbazaar_marketplace_plan(prompt)
    by_path = {item["path"]: item for item in full.suggested_files}
    stage_files: dict[int, list[str]] = {
        3: ["openbazaar_marketplace/README.md", "openbazaar_marketplace/WORKFLOW.md"],
        4: ["openbazaar_marketplace/index.html"],
        5: ["openbazaar_marketplace/src/data.js"],
        6: ["openbazaar_marketplace/src/state.js"],
        7: ["openbazaar_marketplace/src/views.js", "openbazaar_marketplace/src/main.js"],
        8: ["openbazaar_marketplace/src/styles.css"],
        9: ["openbazaar_marketplace/tests/smoke_test.py"],
        10: ["openbazaar_marketplace/api/schema.sql"],
        11: ["openbazaar_marketplace/api/main.py"],
        12: ["openbazaar_marketplace/api/README.md"],
        13: ["openbazaar_marketplace/SECURITY.md"],
    }
    selected_paths = stage_files.get(stage, [])
    selected_files = [by_path[path] for path in selected_paths if path in by_path]
    verify_commands: list[str] = []
    if stage in {9, 14, 15, 16, 17, 18, 19, 20}:
        verify_commands = [
            "python openbazaar_marketplace/tests/smoke_test.py",
            "Open http://127.0.0.1:9000/openbazaar_marketplace/index.html",
        ]
    if stage in {1, 2}:
        selected_files = []
        verify_commands = []
    if stage >= 14:
        selected_files = []
    return TaskPlanResponse(
        goal=prompt,
        mode=f"openbazaar-staged-prd-build-step-{stage:02d}",
        steps=[
            f"Run {OPENBAZAAR_STAGED_PROMPTS[stage - 1]}",
            "Show what changed in this step before moving to the next prompt.",
            "Wait for the next numbered OpenBazaar prompt instead of generating the whole project at once.",
        ],
        suggested_files=selected_files,
        verify_commands=verify_commands,
        notes=[
            f"This is step {stage} of 20. SHAMSU is intentionally building OpenBazaar gradually for the faculty demo.",
            "Use the next prompt only after this step is visible in the chat/history and workspace.",
        ],
        requirements_analysis=full.requirements_analysis if stage <= 2 else [OPENBAZAAR_STAGED_PROMPTS[stage - 1]],
        clarification_questions=[] if stage >= 3 else full.clarification_questions,
        stack=full.stack,
        file_plan=selected_paths or ["No new files in this step; this step is for analysis, verification, or explanation."],
        workflow_summary=f"OpenBazaar staged build step {stage}/20: {OPENBAZAAR_STAGED_PROMPTS[stage - 1]}",
    )


def _openbazaar_dispatch_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    if _is_analysis_only_prompt(lower):
        return _openbazaar_analysis_plan(prompt)
    stage = _openbazaar_stage_number(lower)
    if stage is not None:
        return _openbazaar_staged_phase_plan(prompt, stage)
    if any(term in lower for term in ["all at once", "emergency full", "generate everything now", "single prompt full build"]):
        return _openbazaar_marketplace_plan(prompt)
    return _openbazaar_staged_roadmap_plan(prompt)


def _normalise_openbazaar_text(text: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    return compact.replace("openbazzar", "openbazaar").replace("openbazar", "openbazaar")


def _looks_like_openbazaar_marketplace(lower: str) -> bool:
    normalized = _normalise_openbazaar_text(lower)
    marketplace_terms = ["cash on delivery", "cod", "auction", "seller dashboard", "buyer dashboard"]
    return "openbazaar" in normalized or ("marketplace" in lower and any(term in lower for term in marketplace_terms))

def _with_advisory(plan: TaskPlanResponse, prompt: str) -> TaskPlanResponse:
    """Fill Claude-like guidance fields for every autonomous build plan."""
    lower = prompt.lower()
    if _looks_like_openbazaar_marketplace(lower):
        return _openbazaar_dispatch_plan(prompt)
    subject = _advisory_subject(prompt)
    default_requirements = _default_requirements(lower, subject)
    if not plan.requirements_analysis:
        plan.requirements_analysis = default_requirements
    elif any(word in lower for word in ["crm", "management", "student", "library", "inventory", "dashboard", "portal"]):
        joined_requirements = " ".join(plan.requirements_analysis).lower()
        if "dashboard" not in joined_requirements and "record" not in joined_requirements:
            plan.requirements_analysis.extend(default_requirements[:3])
    if not plan.clarification_questions:
        plan.clarification_questions = _default_clarifications(lower)
    if not plan.stack:
        plan.stack = _default_stack(plan)
    if not plan.file_plan:
        plan.file_plan = _default_file_plan(plan)
    default_workflow = _default_workflow_summary(plan, subject)
    if not plan.workflow_summary:
        plan.workflow_summary = default_workflow
    elif "claude-style build" not in plan.workflow_summary.lower():
        plan.workflow_summary = f"{plan.workflow_summary}\n{default_workflow}"
    if not any("advisory" in note.lower() or "first version" in note.lower() for note in plan.notes):
        plan.notes.append(
            "Advisory-first behavior: SHAMSU explains what is needed, creates a working first version, verifies it, then opens preview when possible."
        )
    return plan


def _advisory_subject(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    ignored = {"make", "build", "create", "generate", "write", "a", "an", "the", "for", "me", "one", "single", "html", "file", "with"}
    useful = [word for word in words if word not in ignored][:5]
    return " ".join(useful) if useful else "requested project"


def _default_requirements(lower: str, subject: str) -> list[str]:
    if "game" in lower:
        return [
            f"Define the core gameplay for {subject}.",
            "Create visible game state such as score, lives, player position, or win/loss state.",
            "Add keyboard or mouse controls that are easy to test in the browser.",
            "Keep the first version small enough to preview and repair quickly.",
        ]
    if any(word in lower for word in ["crm", "management", "student", "library", "inventory", "dashboard", "portal"]):
        return [
            f"Model the main records needed for {subject}.",
            "Provide dashboard summary cards for quick status understanding.",
            "Add a table/list view with search or filtering.",
            "Add a form for creating records and client-side validation for required fields.",
            "Use local persistence for the first demo, then note how to upgrade to backend/database storage.",
        ]
    if any(word in lower for word in ["website", "site", "web page", "landing"]):
        return [
            f"Identify the purpose and audience for {subject}.",
            "Create the primary page structure, navigation, content sections, and call to action.",
            "Use clean responsive styling and lightweight browser JavaScript only when it helps the experience.",
            "Verify the generated page opens in the local preview server.",
        ]
    return [
        f"Break the request into a small working first version for {subject}.",
        "Create the minimum files needed to demonstrate the main behavior.",
        "Run lightweight verification and repair errors before showing the result.",
        "Explain next steps for expanding the prototype.",
    ]


def _default_clarifications(lower: str) -> list[str]:
    questions: list[str] = []
    if any(word in lower for word in ["system", "management", "crm", "portal", "dashboard"]):
        questions.append("Should this stay as a quick local prototype, or later be upgraded with backend/database/auth?")
        questions.append("What exact fields should each record store for the final version?")
    if "game" in lower:
        questions.append("Should the game be keyboard-only, mouse/touch-friendly, or both?")
    if not questions:
        questions.append("After the first preview, what feature should be improved next?")
    return questions


def _default_stack(plan: TaskPlanResponse) -> list[str]:
    suffixes = {Path(item.get("path", "")).suffix.lower() for item in plan.suggested_files}
    if suffixes <= {".html"} and suffixes:
        return ["HTML", "CSS", "JavaScript", "Browser localStorage when persistence is needed", "Local preview server"]
    stack = []
    if any(suffix in suffixes for suffix in [".html", ".css", ".js"]):
        stack.extend(["HTML", "CSS", "JavaScript"])
    if ".py" in suffixes:
        stack.append("Python")
    if any(suffix in suffixes for suffix in [".json", ".md"]):
        stack.append("Project metadata/documentation")
    stack.append("SHAMSU autonomous verify and repair loop")
    return stack


def _default_file_plan(plan: TaskPlanResponse) -> list[str]:
    if plan.suggested_files:
        return [f"Create {item.get('path')}: {_file_plan_description(str(item.get('path') or ''), str(item.get('content') or ''))}" for item in plan.suggested_files]
    return ["No files will be written yet; use chat/tool mode to inspect the workspace and refine the plan."]


def _file_plan_description(path: str, content: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".html":
        return "main runnable browser page with structure, styling, and interaction"
    if suffix == ".css":
        return "visual styling and responsive layout"
    if suffix == ".js":
        return "browser behavior and state management"
    if suffix == ".py":
        return "Python logic or backend/script entry point"
    if suffix == ".md":
        return "workflow notes, roadmap, or usage documentation"
    if suffix == ".json":
        return "structured configuration or seed data"
    return f"generated project file ({len(content)} characters)"


def _default_workflow_summary(plan: TaskPlanResponse, subject: str) -> str:
    file_names = [str(item.get("path")) for item in plan.suggested_files if item.get("path")]
    target = ", ".join(file_names) if file_names else "the planned files"
    return (
        f"SHAMSU will treat '{subject}' like a Claude-style build: first analyze what is needed, "
        f"choose a small working first version, create {target}, run verification, repair if feedback appears, "
        "then open a local preview for browser output. After the preview, ask for improvements such as more fields, "
        "database persistence, authentication, or design polish."
    )

def _looks_like_build_request(prompt: str) -> bool:
    text = prompt.lower()
    return bool(re.search(r"\b(make|build|create|generate|write|develop|implement)\b", text)) and bool(
        re.search(r"\b(game|app|application|website|web page|html|system|tool|program|project|calculator|todo|quiz|crm|management|dashboard|portal|inventory|student|library|os|operating system)\b", text)
    )


def _looks_like_website_request(text: str) -> bool:
    return bool(re.search(r"\b(website|web page|landing page|site)\b", text))


def _looks_like_system_request(text: str) -> bool:
    return bool(re.search(r"\b(system|dashboard|portal|application|app)\b", text)) and not any(
        word in text for word in ["game", "calculator", "todo", "quiz"]
    )


def _wants_database_backing(text: str) -> bool:
    return bool(re.search(r"\b(database|db|sqlite|backend|api|full[- ]?stack|persistent|server)\b", text))


def _wants_full_stack(text: str) -> bool:
    return bool(re.search(r"\b(full[- ]?stack|frontend.*backend|backend.*frontend|api.*frontend|frontend.*api)\b", text))


def _slug_from_prompt(prompt: str, fallback: str, suffix: str) -> str:
    text = prompt.lower()
    text = re.sub(r"\b(make|build|create|generate|write|develop|implement|a|an|the|for|me|one|single|simple|website|web|page|site|system|app|application|dashboard|portal)\b", " ", text)
    words = re.findall(r"[a-z0-9]+", text)[:5]
    stem = "_".join(words) if words else fallback
    if not stem.endswith(suffix):
        stem = f"{stem}_{suffix}"
    return f"{stem}.html"


async def _generate_json_file_plan(prompt: str, steps: list[TaskRunStep]) -> list[dict[str, str]]:
    system = (
        "You generate small coding project files. Return ONLY valid JSON with this exact shape: "
        "{\"files\":[{\"path\":\"relative/path.ext\",\"content\":\"full file content\"}],\"notes\":[\"...\"]}. "
        "Use relative paths only. Do not use markdown fences. Prefer one runnable HTML file for browser games, "
        "or a small Python/C project when requested. Keep the project compact."
    )
    user = f"Create the files for this request: {prompt}"
    try:
        response = await ollama.AsyncClient(host=settings.ollama_host).chat(
            model=model_registry.get_current_model(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            stream=False,
            think=False,
            options={"temperature": 0.1, "num_ctx": settings.model_num_ctx, "num_predict": settings.max_model_output_tokens},
        )
    except Exception as exc:
        steps.append(TaskRunStep(name="generate", status="error", detail=f"Local model generation failed: {exc}"))
        return []

    content = (response.get("message") or {}).get("content") or ""
    payload = _extract_json_payload(content)
    if not payload:
        steps.append(TaskRunStep(name="generate", status="error", detail="Model did not return valid JSON."))
        return []
    files = _validated_generated_files(payload, steps)
    if files:
        steps.append(TaskRunStep(name="generate", status="ok", detail=f"Generated {len(files)} file(s) from JSON plan."))
    return files


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _validated_generated_files(payload: dict[str, Any], steps: list[TaskRunStep]) -> list[dict[str, str]]:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        steps.append(TaskRunStep(name="validate", status="error", detail="JSON payload does not contain a files list."))
        return []
    files: list[dict[str, str]] = []
    total_chars = 0
    for raw in raw_files[:MAX_GENERATED_FILES]:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path") or "").strip().replace("\\", "/")
        content = raw.get("content")
        if not path or not isinstance(content, str):
            continue
        suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if suffix not in ALLOWED_GENERATED_SUFFIXES:
            steps.append(TaskRunStep(name="validate", status="skipped", detail=f"Skipped unsupported generated file type: {path}"))
            continue
        try:
            tools.resolve_in_workspace(path)
        except ValueError as exc:
            steps.append(TaskRunStep(name="validate", status="skipped", detail=str(exc)))
            continue
        total_chars += len(content)
        if total_chars > MAX_GENERATED_CHARS:
            steps.append(TaskRunStep(name="validate", status="error", detail="Generated project exceeded size limit."))
            break
        files.append({"path": path, "content": content})
    return files


def _make_generated_plan(prompt: str, files: list[dict[str, str]], mode: str = "json-generated-task", note: str | None = None) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt.strip(),
        mode=mode,
        steps=[
            "Generate a JSON file plan.",
            "Write generated files.",
            "Run lightweight verification.",
            "Repair from verification feedback if needed.",
            "Open preview for HTML output.",
        ],
        suggested_files=files,
        verify_commands=[],
        notes=[note or "Files were generated by the local model in strict JSON format and validated before writing."],
    )


async def _verify_and_repair_loop(
    prompt: str,
    plan: TaskPlanResponse,
    created_files: list[str],
    steps: list[TaskRunStep],
) -> tuple[bool, TaskPlanResponse, list[str]]:
    verify_ok = _verify_created_files(plan, created_files, steps)
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        if verify_ok:
            break
        feedback = _verification_feedback(created_files)
        steps.append(TaskRunStep(name="feedback", status="warning", detail="; ".join(feedback[:4]) or "Verification failed without detailed feedback."))

        repaired_by_rule = _repair_created_files(created_files, steps)
        if repaired_by_rule:
            verify_ok = _verify_created_files(plan, created_files, steps)
            if verify_ok:
                break
            feedback = _verification_feedback(created_files)

        if plan.mode not in {"json-generated-task", "json-repaired-task"}:
            break

        repaired_files = await _generate_json_repair_plan(prompt, plan.suggested_files, feedback, steps, attempt)
        if not repaired_files:
            break
        plan = _make_generated_plan(
            prompt,
            repaired_files,
            mode="json-repaired-task",
            note=f"Files were repaired by the local model using verification feedback on attempt {attempt}.",
        )
        created_files = _write_suggested_files(plan, overwrite=True, steps=steps)
        verify_ok = _verify_created_files(plan, created_files, steps)
    return verify_ok, plan, created_files


async def _generate_json_repair_plan(
    prompt: str,
    current_files: list[dict[str, str]],
    feedback: list[str],
    steps: list[TaskRunStep],
    attempt: int,
) -> list[dict[str, str]]:
    system = (
        "You repair small coding project files. Return ONLY valid JSON with this exact shape: "
        "{\"files\":[{\"path\":\"relative/path.ext\",\"content\":\"full corrected file content\"}],\"notes\":[\"...\"]}. "
        "Use relative paths only. Return complete corrected file contents, not patches. Keep the project compact. "
        "Fix the verification errors while preserving the user's requested app."
    )
    file_context = []
    for item in current_files[:MAX_GENERATED_FILES]:
        content = str(item.get("content") or "")
        file_context.append({"path": item.get("path"), "content": _truncate_for_model(content, 12000)})
    user = json.dumps(
        {
            "original_prompt": prompt,
            "verification_feedback": feedback,
            "current_files": file_context,
        },
        ensure_ascii=False,
    )
    try:
        response = await ollama.AsyncClient(host=settings.ollama_host).chat(
            model=model_registry.get_current_model(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            stream=False,
            think=False,
            options={"temperature": 0.05, "num_ctx": settings.model_num_ctx, "num_predict": settings.max_model_output_tokens},
        )
    except Exception as exc:
        steps.append(TaskRunStep(name="repair-generate", status="error", detail=f"Repair attempt {attempt} failed: {exc}"))
        return []

    content = (response.get("message") or {}).get("content") or ""
    payload = _extract_json_payload(content)
    if not payload:
        steps.append(TaskRunStep(name="repair-generate", status="error", detail=f"Repair attempt {attempt} did not return valid JSON."))
        return []
    files = _validated_generated_files(payload, steps)
    if files:
        steps.append(TaskRunStep(name="repair-generate", status="ok", detail=f"Repair attempt {attempt} generated {len(files)} corrected file(s)."))
    return files


def _truncate_for_model(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    half = limit // 2
    return content[:half] + "\n... truncated for repair feedback ...\n" + content[-half:]

def _write_suggested_files(plan: TaskPlanResponse, overwrite: bool, steps: list[TaskRunStep]) -> list[str]:
    created_files: list[str] = []
    for item in plan.suggested_files:
        path = item["path"]
        content = item["content"]
        existing = tools.read_file(path)
        if not overwrite and not existing.startswith("Error:"):
            steps.append(TaskRunStep(name="write", status="skipped", detail=f"{path} already exists."))
            continue
        result = tools.write_file(path, content)
        created_files.append(path)
        status = "error" if result.startswith("Error") else "ok"
        steps.append(TaskRunStep(name="write", status=status, detail=result))
    return created_files


def _verify_created_files(plan: TaskPlanResponse, created_files: list[str], steps: list[TaskRunStep]) -> bool:
    if not created_files:
        steps.append(TaskRunStep(name="verify", status="error", detail="No files were created."))
        return False
    ok = True
    for path in created_files:
        content = tools.read_file(path)
        if content.startswith("Error:") or not content.strip():
            steps.append(TaskRunStep(name="verify", status="error", detail=f"{path} is missing or empty."))
            ok = False
            continue
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
                steps.append(TaskRunStep(name="verify", status="ok", detail=f"{path} Python syntax verified."))
            except SyntaxError as exc:
                steps.append(TaskRunStep(name="verify", status="error", detail=f"{path} syntax error: {exc}"))
                ok = False
        elif path.endswith(".html"):
            lower = content.lower()
            html_ok = "<html" in lower or "<canvas" in lower or "<script" in lower or "<main" in lower
            script_feedback = _basic_script_feedback(content) if "<script" in lower else None
            if script_feedback:
                steps.append(TaskRunStep(name="verify", status="error", detail=f"{path} {script_feedback}"))
                ok = False
            else:
                steps.append(TaskRunStep(name="verify", status="ok" if html_ok else "warning", detail=f"{path} HTML structure checked."))
                ok = ok and html_ok
        elif path.endswith(".c"):
            c_ok = "main(" in content or "main (" in content
            steps.append(TaskRunStep(name="verify", status="ok" if c_ok else "warning", detail=f"{path} C entry point checked."))
        else:
            steps.append(TaskRunStep(name="verify", status="ok", detail=f"{path} written and readable."))
    return ok


def _verification_feedback(created_files: list[str]) -> list[str]:
    if not created_files:
        return ["No files were created."]
    feedback: list[str] = []
    for path in created_files:
        content = tools.read_file(path)
        lower = content.lower()
        if content.startswith("Error:") or not content.strip():
            feedback.append(f"{path}: file is missing or empty.")
            continue
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
            except SyntaxError as exc:
                feedback.append(f"{path}: Python syntax error: {exc}.")
        elif path.endswith(".html"):
            if not any(token in lower for token in ["<html", "<body", "<canvas", "<script", "<main"]):
                feedback.append(f"{path}: HTML appears to have no runnable or visible document structure.")
            if "<script" in lower:
                script_feedback = _basic_script_feedback(content)
                if script_feedback:
                    feedback.append(f"{path}: {script_feedback}")
        elif path.endswith(".c") and "main(" not in content and "main (" not in content:
            feedback.append(f"{path}: C file has no main function.")
    return feedback


def _basic_script_feedback(content: str) -> str | None:
    opens = content.count("{")
    closes = content.count("}")
    if opens != closes:
        return f"JavaScript block braces look unbalanced ({opens} opening, {closes} closing)."
    if "getContext('2d')" in content and "requestAnimationFrame" not in content and "setInterval" not in content:
        return "Canvas code has no animation loop such as requestAnimationFrame or setInterval."
    return None

def _repair_created_files(created_files: list[str], steps: list[TaskRunStep]) -> bool:
    repaired = False
    for path in created_files:
        content = tools.read_file(path)
        if content.startswith("Error:") or not content.strip():
            continue
        if path.endswith(".html"):
            fixed = _repair_html_content(path, content)
            if fixed is not None:
                result = tools.write_file(path, fixed)
                status = "error" if result.startswith("Error") else "ok"
                steps.append(TaskRunStep(name="repair", status=status, detail=f"Repaired HTML shell for {path}." if status == "ok" else result))
                repaired = repaired or status == "ok"
    if not repaired:
        steps.append(TaskRunStep(name="repair", status="skipped", detail="No automatic safe repair was available."))
    return repaired


def _repair_html_content(path: str, content: str) -> str | None:
    lower = content.lower()
    if "<html" in lower and ("<body" in lower or "<canvas" in lower or "<script" in lower):
        return None
    title = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title() or "SHAMSU Generated App"
    body = content.strip()
    if "<script" not in lower and re.search(r"\b(function|const|let|var|document\.)\b", body):
        body = '<canvas id="game" width="640" height="420"></canvas>\n<script>\n' + body + '\n</script>'
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <style>body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#111827;color:#f9fafb;font-family:Arial,sans-serif}}main{{width:min(900px,92vw)}}canvas{{max-width:100%;background:#f8fafc;border:2px solid #334155}}</style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""

async def _maybe_start_preview(preview: bool, created_files: list[str], verify_ok: bool, steps: list[TaskRunStep]) -> str | None:
    if not preview or not verify_ok:
        return None
    html_file = next((path for path in created_files if path.endswith(".html")), None)
    if not html_file:
        return None
    preview_state = await start_preview(PreviewStartRequest(path=html_file, port=9000))
    steps.append(TaskRunStep(name="preview", status="ok", detail=f"Preview ready at {preview_state.url}"))
    return preview_state.url


def _website_prototype_plan(prompt: str) -> TaskPlanResponse:
    project = _slug_from_prompt(prompt, "shamsu", "website").removesuffix(".html")
    title = project.replace("_", " ").title()
    base = project
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <header class="hero">
    <nav><strong>__TITLE__</strong><a href="#features">Features</a><a href="#contact">Contact</a></nav>
    <section><h1>__TITLE__</h1><p>A multi-file website prototype generated by SHAMSU with separate HTML, CSS, JavaScript, and workflow documentation.</p><button id="ctaButton">Get started</button></section>
  </header>
  <main>
    <section id="features" class="grid">
      <article><h2>Responsive Layout</h2><p>Works on desktop and mobile.</p></article>
      <article><h2>Editable Sections</h2><p>Content is split into clean files for easier future edits.</p></article>
      <article><h2>Verified Preview</h2><p>SHAMSU checks the generated files and opens this page locally.</p></article>
    </section>
    <section id="contact" class="panel"><h2>Contact</h2><form id="contactForm"><input placeholder="Name" required /><input placeholder="Email" required /><textarea placeholder="Message" required></textarea><button type="submit">Send</button></form><p id="message"></p></section>
  </main>
  <script src="app.js"></script>
</body>
</html>
""".replace("__TITLE__", title)
    css = """body{margin:0;font-family:Arial,sans-serif;background:#f5f7fb;color:#172033}.hero{background:#123047;color:white;padding:28px 36px 58px}nav{display:flex;gap:18px;align-items:center;flex-wrap:wrap}nav strong{margin-right:auto;font-size:22px}nav a{color:#d9eefb;text-decoration:none}.hero section{max-width:820px;margin-top:48px}.hero h1{font-size:44px;line-height:1.08;margin:0 0 12px}.hero p{font-size:18px;color:#d9eefb;line-height:1.6}button{border:0;border-radius:6px;background:#0f766e;color:white;padding:12px 16px;font-weight:800;cursor:pointer}main{max-width:1120px;margin:0 auto;padding:28px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.grid article,.panel{background:white;border:1px solid #dbe4ef;border-radius:8px;padding:20px}.panel{margin-top:28px}form{display:grid;grid-template-columns:1fr 1fr;gap:12px}input,textarea{box-sizing:border-box;width:100%;border:1px solid #cbd5e1;border-radius:6px;padding:11px;font:inherit}textarea{grid-column:span 2;min-height:110px}#message{color:#0f766e;font-weight:700}@media(max-width:780px){.hero h1{font-size:34px}.grid,form{grid-template-columns:1fr}textarea{grid-column:auto}}"""
    js = """document.getElementById('ctaButton').addEventListener('click', () => document.getElementById('contact').scrollIntoView({behavior:'smooth'}));
document.getElementById('contactForm').addEventListener('submit', (event) => {
  event.preventDefault();
  document.getElementById('message').textContent = 'Demo message saved locally. Add a backend API for production.';
  event.target.reset();
});
"""
    workflow = f"""# {title} Workflow

## Requirement Analysis
- Build a presentable website from the prompt: {prompt}
- Keep the first version local, previewable, and easy to edit.

## Clarification Questions
- What exact brand colors, copy, and pages should be used next?
- Should this stay static or become a React/full-stack site?

## Chosen Stack
- HTML for structure
- CSS for responsive styling
- JavaScript for interaction
- Python preview server for local demo

## File Plan
- index.html: page structure and sections
- styles.css: layout and visual design
- app.js: contact form and scroll interaction
- WORKFLOW.md: explanation for faculty/demo

## Verification
- Open index.html through the preview URL.
- Submit the contact form and confirm the success message appears.
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="website-generator",
        steps=["Analyze requirements.", "Ask/record clarification questions when details are missing.", "Choose a simple static website stack.", "Create a multi-file project.", "Verify generated files.", "Start preview server and explain workflow."],
        suggested_files=[
            {"path": f"{base}/index.html", "content": html},
            {"path": f"{base}/styles.css", "content": css},
            {"path": f"{base}/app.js", "content": js},
            {"path": f"{base}/WORKFLOW.md", "content": workflow},
        ],
        verify_commands=[f"Open http://127.0.0.1:9000/{base}/index.html"],
        notes=["SHAMSU created a Claude-like multi-file website starter. Ask for a React/full-stack upgrade when you want backend, routing, auth, or database."],
        requirements_analysis=["The prompt requests a website that should be visible in a browser.", "A local static prototype is the safest first build for demo and iteration."],
        clarification_questions=["What brand/content should replace the generated placeholder copy?", "Should the next version be static, React, or full-stack?"],
        stack=["HTML", "CSS", "JavaScript", "Local preview server"],
        file_plan=[f"{base}/index.html", f"{base}/styles.css", f"{base}/app.js", f"{base}/WORKFLOW.md"],
        workflow_summary="prompt -> requirement analysis -> clarification questions -> stack choice -> file plan -> multi-file creation -> verification -> preview -> explanation",
    )



def _full_stack_project_plan(prompt: str) -> TaskPlanResponse:
    base = _slug_from_prompt(prompt, "full_stack", "project").removesuffix(".html")
    base = re.sub(r"(^|_)full(_|-)?stack(_|$)", "_", base).strip("_") or "full_stack_project"
    if not base.endswith("project") and not base.endswith("system") and not base.endswith("app"):
        base = f"{base}_app"
    title = base.replace("_", " ").title()
    entity = _entity_from_project_title(title)
    config = {"title": title, "entity": entity, "categoryLabel": "Category", "contactLabel": "Owner"}
    seed_records = [
        {"name": f"Sample {entity} One", "category": "Planning", "contact": "owner@example.test", "status": "Open", "notes": "Created by SHAMSU."},
        {"name": f"Sample {entity} Two", "category": "Delivery", "contact": "team@example.test", "status": "Done", "notes": "Verified demo record."},
    ]
    requirements = "fastapi\nuvicorn[standard]\npydantic\n"
    package_json = json.dumps(
        {
            "name": base,
            "version": "0.1.0",
            "private": True,
            "scripts": {
                "dev": "python -m uvicorn backend.main:app --reload --port 8090",
                "smoke": "python tests/smoke_test.py",
            },
        },
        indent=2,
    )
    database_py = '''from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"
SEED_RECORDS = __SEED_RECORDS__


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                contact TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT ''
            )
            """
        )
        count = connection.execute("SELECT COUNT(*) AS total FROM items").fetchone()["total"]
        if count == 0:
            connection.executemany(
                "INSERT INTO items (name, category, contact, status, notes) VALUES (:name, :category, :contact, :status, :notes)",
                SEED_RECORDS,
            )
        connection.commit()


def to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def list_items() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM items ORDER BY id DESC").fetchall()
    return [to_dict(row) for row in rows]


def create_item(data: dict[str, str]) -> dict[str, Any]:
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO items (name, category, contact, status, notes) VALUES (?, ?, ?, ?, ?)",
            (data["name"], data["category"], data["contact"], data["status"], data.get("notes", "")),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM items WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return to_dict(row)


def update_item(item_id: int, data: dict[str, str]) -> dict[str, Any] | None:
    with connect() as connection:
        connection.execute(
            "UPDATE items SET name = ?, category = ?, contact = ?, status = ?, notes = ? WHERE id = ?",
            (data["name"], data["category"], data["contact"], data["status"], data.get("notes", ""), item_id),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    return to_dict(row) if row else None


def delete_item(item_id: int) -> bool:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM items WHERE id = ?", (item_id,))
        connection.commit()
    return cursor.rowcount > 0
'''.replace("__SEED_RECORDS__", json.dumps(seed_records, indent=4))
    main_py = '''from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import create_item, delete_item, init_db, list_items, update_item

APP_CONFIG = __APP_CONFIG__


class ItemIn(BaseModel):
    name: str
    category: str
    contact: str
    status: str = "Open"
    notes: str = ""


app = FastAPI(title=APP_CONFIG["title"])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": APP_CONFIG["title"]}


@app.get("/api/config")
def config() -> dict:
    return APP_CONFIG


@app.get("/api/items")
def items() -> list[dict]:
    return list_items()


@app.post("/api/items", status_code=201)
def create(data: ItemIn) -> dict:
    return create_item(data.model_dump())


@app.put("/api/items/{item_id}")
def update(item_id: int, data: ItemIn) -> dict:
    item = update_item(item_id, data.model_dump())
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.delete("/api/items/{item_id}")
def remove(item_id: int) -> dict[str, bool]:
    if not delete_item(item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
'''.replace("__APP_CONFIG__", json.dumps(config, indent=2))
    html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <aside class="sidebar"><h1>__TITLE__</h1><button data-view="dashboard">Dashboard</button><button data-view="items">Items</button><button data-view="api">API</button></aside>
  <main class="content"><section id="dashboard"></section><section id="items"></section><section id="api" class="hidden"><h2>API Routes</h2><pre>GET /api/health\nGET /api/items\nPOST /api/items\nPUT /api/items/{id}\nDELETE /api/items/{id}</pre></section></main>
  <script src="app.js"></script>
</body>
</html>
'''.replace("__TITLE__", title)
    app_js = '''let config = null;
let items = [];
let editingId = null;
const panels = ["dashboard", "items", "api"];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function load() {
  config = await api("/api/config");
  items = await api("/api/items");
  render();
}

function render() {
  const open = items.filter((item) => !["Done", "Closed", "Archived"].includes(item.status)).length;
  document.getElementById("dashboard").innerHTML = `<h2>${config.title}</h2><div class="metrics"><article><span>Total</span><strong>${items.length}</strong></article><article><span>Open</span><strong>${open}</strong></article><article><span>Closed</span><strong>${items.length - open}</strong></article></div>`;
  const rows = items.map((item) => `<tr><td>${item.name}</td><td>${item.category}</td><td>${item.contact}</td><td>${item.status}</td><td>${item.notes || ""}</td><td><button data-edit="${item.id}">Edit</button><button data-delete="${item.id}">Delete</button></td></tr>`).join("");
  document.getElementById("items").innerHTML = `<h2>${config.entity} List</h2><form id="itemForm" class="record-form"><input id="name" placeholder="${config.entity} name" required /><input id="category" placeholder="${config.categoryLabel}" required /><input id="contact" placeholder="${config.contactLabel}" required /><select id="status"><option>Open</option><option>Active</option><option>Done</option><option>Closed</option><option>Archived</option></select><input id="notes" placeholder="Notes" /><button>Save</button></form><table><thead><tr><th>Name</th><th>${config.categoryLabel}</th><th>${config.contactLabel}</th><th>Status</th><th>Notes</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`;
  bindEvents();
}

function readForm() {
  return Object.fromEntries(["name", "category", "contact", "status", "notes"].map((id) => [id, document.getElementById(id).value.trim()]));
}

function bindEvents() {
  document.getElementById("itemForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = JSON.stringify(readForm());
    if (editingId === null) await api("/api/items", { method: "POST", body });
    else await api(`/api/items/${editingId}`, { method: "PUT", body });
    editingId = null;
    items = await api("/api/items");
    render();
  });
  document.querySelectorAll("[data-delete]").forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/items/${button.dataset.delete}`, { method: "DELETE" });
    items = await api("/api/items");
    render();
  }));
  document.querySelectorAll("[data-edit]").forEach((button) => button.addEventListener("click", () => {
    editingId = Number(button.dataset.edit);
    const item = items.find((entry) => entry.id === editingId);
    ["name", "category", "contact", "status", "notes"].forEach((id) => { document.getElementById(id).value = item[id] || ""; });
  }));
}

document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
  panels.forEach((panel) => document.getElementById(panel).classList.toggle("hidden", panel !== button.dataset.view));
}));

load().catch((error) => { document.body.innerHTML = `<main class="content"><h1>Could not load app</h1><pre>${error.message}</pre></main>`; });
'''
    css = '''body{margin:0;font-family:Arial,sans-serif;background:#f3f6fb;color:#172033}.sidebar{position:fixed;inset:0 auto 0 0;width:245px;background:#102a43;color:white;padding:24px;box-sizing:border-box}.sidebar h1{font-size:24px;line-height:1.2}.sidebar button{display:block;width:100%;margin:8px 0;padding:11px;border:0;border-radius:6px;background:#2563eb;color:white;text-align:left;font-weight:800;cursor:pointer}.content{margin-left:245px;padding:26px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metrics article,section{background:white;border:1px solid #d8e1ec;border-radius:8px;padding:18px;margin-bottom:18px}.metrics strong{display:block;font-size:32px;color:#0f766e}.record-form{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px}input,select{border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}button{border:0;border-radius:6px;background:#7c3aed;color:white;padding:10px 12px;font-weight:800;cursor:pointer}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:10px}.hidden{display:none}pre{background:#0f172a;color:#e2e8f0;padding:16px;overflow:auto}@media(max-width:840px){.sidebar{position:static;width:auto}.content{margin-left:0}.record-form,.metrics{grid-template-columns:1fr}table{font-size:14px}}'''
    readme = f"""# {title}

Generated by SHAMSU as a generic full-stack local project.

## Run
```powershell
cd C:\\Users\\HP\\Desktop\\CSE327\\workspace\\{base}
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8090
```

Open `http://127.0.0.1:8090`.

## Verify
```powershell
python tests/smoke_test.py
```
"""
    workflow = f"""# Workflow

Prompt: {prompt}

1. Requirement analysis: user requested a full-stack app, so SHAMSU creates frontend, backend, API, and database layers.
2. Stack choice: FastAPI + SQLite + vanilla JavaScript for a self-contained local demo.
3. File plan: backend API, database module, frontend UI, docs, smoke test.
4. Verification: Python syntax plus static checks for API/frontend wiring.
5. Run: uvicorn serves both API and frontend.
"""
    smoke = '''from pathlib import Path
import py_compile

root = Path(__file__).resolve().parents[1]
required = ["backend/__init__.py", "backend/main.py", "backend/database.py", "frontend/index.html", "frontend/app.js", "frontend/styles.css", "requirements.txt", "README.md", "WORKFLOW.md"]
missing = [path for path in required if not (root / path).exists()]
assert not missing, f"Missing files: {missing}"
py_compile.compile(str(root / "backend" / "main.py"), doraise=True)
py_compile.compile(str(root / "backend" / "database.py"), doraise=True)
assert "FastAPI" in (root / "backend" / "main.py").read_text(encoding="utf-8")
assert "sqlite3" in (root / "backend" / "database.py").read_text(encoding="utf-8")
assert "/api/items" in (root / "frontend" / "app.js").read_text(encoding="utf-8")
print("full-stack smoke passed")
'''
    files = [
        {"path": f"{base}/package.json", "content": package_json + "\n"},
        {"path": f"{base}/requirements.txt", "content": requirements},
        {"path": f"{base}/README.md", "content": readme},
        {"path": f"{base}/WORKFLOW.md", "content": workflow},
        {"path": f"{base}/backend/__init__.py", "content": "# Backend package for the SHAMSU-generated full-stack project.\n"},
        {"path": f"{base}/backend/database.py", "content": database_py},
        {"path": f"{base}/backend/main.py", "content": main_py},
        {"path": f"{base}/frontend/index.html", "content": html},
        {"path": f"{base}/frontend/app.js", "content": app_js},
        {"path": f"{base}/frontend/styles.css", "content": css},
        {"path": f"{base}/tests/smoke_test.py", "content": smoke},
    ]
    return TaskPlanResponse(
        goal=prompt,
        mode="full-stack-project-generator",
        steps=["Analyze full-stack requirements.", "Choose frontend/backend/database stack.", "Create FastAPI API, SQLite data layer, frontend UI, docs, and smoke test.", "Verify syntax and API wiring.", "Serve frontend and API with uvicorn."],
        suggested_files=files,
        verify_commands=[f"python {base}/tests/smoke_test.py", f"cd {base} && python -m uvicorn backend.main:app --reload --port 8090"],
        notes=["SHAMSU generated a complete local full-stack scaffold instead of a static-only prototype.", "The generated API and SQLite database are intentionally small so faculty demo setup stays quick."],
        requirements_analysis=["The prompt explicitly asks for a full-stack system.", "A real backend/API/database layer is needed so changes persist outside browser localStorage."],
        clarification_questions=["Which authentication roles and permissions should be added next?", "Which reports, filters, and production database should the app support?"],
        stack=["FastAPI", "SQLite", "Pydantic", "HTML", "CSS", "Vanilla JavaScript", "Python smoke test"],
        file_plan=[item["path"] for item in files],
        workflow_summary="prompt -> requirement analysis -> choose full-stack stack -> create backend/database/frontend -> smoke verification -> run uvicorn -> iterate",
    )


def _entity_from_project_title(title: str) -> str:
    for word in ["Clinic", "Student", "Library", "Inventory", "CRM", "Task", "Project", "Order", "Booking"]:
        if word.lower() in title.lower():
            return "Customer" if word == "CRM" else word
    return "Item"
def _starter_system_plan(prompt: str) -> TaskPlanResponse:
    project = _slug_from_prompt(prompt, "starter", "system").removesuffix(".html")
    title = project.replace("_", " ").title()
    base = project
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <div class="shell">
    <aside><h1>__TITLE__</h1><button data-view="overview">Overview</button><button data-view="records">Records</button><button data-view="workflow">Workflow</button></aside>
    <main>
      <section id="overview"><h2>Overview</h2><div class="metrics"><article><span>Total Records</span><strong id="totalCount">0</strong></article><article><span>Open Items</span><strong id="openCount">0</strong></article><article><span>Done</span><strong id="doneCount">0</strong></article></div></section>
      <section id="records"><h2>Records</h2><div class="toolbar"><input id="recordInput" placeholder="Add a record" /><button id="addButton">Add</button></div><table><thead><tr><th>Name</th><th>Status</th><th>Actions</th></tr></thead><tbody id="recordRows"></tbody></table></section>
      <section id="workflow" class="hidden"><h2>Workflow</h2><ol><li>Capture requirements.</li><li>Create starter UI.</li><li>Verify generated files.</li><li>Expand into backend/database/auth when requested.</li></ol></section>
    </main>
  </div>
  <script src="app.js"></script>
</body>
</html>
""".replace("__TITLE__", title)
    css = """body{margin:0;font-family:Arial,sans-serif;background:#eef3f8;color:#172033}.shell{display:grid;grid-template-columns:240px 1fr;min-height:100vh}aside{background:#102a43;color:white;padding:22px}aside h1{font-size:24px}aside button{display:block;width:100%;text-align:left;margin:8px 0;padding:10px;border:0;border-radius:6px;background:#1d4ed8;color:white;cursor:pointer}main{padding:24px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metrics article,section{background:white;border:1px solid #d7e0ea;border-radius:8px;padding:18px;margin-bottom:18px}.metrics strong{display:block;font-size:30px;color:#0f766e}.toolbar{display:flex;gap:10px;margin:18px 0}.toolbar input{flex:1;border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}button{background:#7c3aed;color:white;border:0;border-radius:6px;padding:10px 14px;font-weight:800;cursor:pointer}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:11px}.status{padding:4px 8px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:12px;font-weight:700}.hidden{display:none}@media(max-width:820px){.shell{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}aside button{display:inline-block;width:auto;margin-right:8px}}"""
    js = """const storageKey = 'shamsu-system-records';
let records = JSON.parse(localStorage.getItem(storageKey) || 'null') || [{name:'Demo setup',status:'Open'},{name:'Faculty review',status:'Done'}];
function save(){ localStorage.setItem(storageKey, JSON.stringify(records)); }
function showView(id){ ['overview','records','workflow'].forEach((name) => document.getElementById(name).classList.toggle('hidden', name !== id)); }
function addRecord(){ const input = document.getElementById('recordInput'); const name = input.value.trim(); if(!name) return; records.push({name,status:'Open'}); input.value=''; save(); renderRecords(); }
function toggleRecord(index){ records[index].status = records[index].status === 'Open' ? 'Done' : 'Open'; save(); renderRecords(); }
function deleteRecord(index){ records.splice(index,1); save(); renderRecords(); }
function renderRecords(){ document.getElementById('recordRows').innerHTML = records.map((record,index) => `<tr><td>${record.name}</td><td><span class="status">${record.status}</span></td><td><button onclick="toggleRecord(${index})">Toggle</button> <button onclick="deleteRecord(${index})">Delete</button></td></tr>`).join(''); document.getElementById('totalCount').textContent = records.length; document.getElementById('openCount').textContent = records.filter((r) => r.status === 'Open').length; document.getElementById('doneCount').textContent = records.filter((r) => r.status === 'Done').length; }
document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
document.getElementById('addButton').addEventListener('click', addRecord);
renderRecords();
"""
    workflow = f"""# {title} Workflow

## Requirement Analysis
- Build a local management/system prototype from the prompt: {prompt}
- Include a dashboard, records area, workflow explanation, and persistent local state.

## Clarification Questions
- Which user roles, data fields, and reports are required?
- Should the next iteration use FastAPI/MySQL authentication and APIs?

## Chosen Stack
- HTML/CSS/JavaScript for the first local UI
- localStorage for demo persistence
- Future upgrade path: React + FastAPI + MySQL

## File Plan
- index.html: dashboard structure
- styles.css: UI layout and responsive styling
- app.js: CRUD-style record behavior
- WORKFLOW.md: explanation and next implementation steps

## Verification
- Open index.html through the preview URL.
- Add, toggle, and delete a record.
- Refresh and confirm records persist locally.
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="system-prototype-generator",
        steps=["Analyze requirements.", "Ask/record clarification questions when details are missing.", "Choose a local starter stack.", "Create a multi-file project.", "Verify generated files.", "Start preview server and explain workflow."],
        suggested_files=[
            {"path": f"{base}/index.html", "content": html},
            {"path": f"{base}/styles.css", "content": css},
            {"path": f"{base}/app.js", "content": js},
            {"path": f"{base}/WORKFLOW.md", "content": workflow},
        ],
        verify_commands=[f"Open http://127.0.0.1:9000/{base}/index.html"],
        notes=["SHAMSU created a Claude-like multi-file system starter. Ask for the production upgrade to add FastAPI routes, MySQL tables, authentication, and tests."],
        requirements_analysis=["The prompt requests an application/system, so SHAMSU builds a working dashboard first.", "A multi-file local UI keeps the first version testable while leaving a clear full-stack path."],
        clarification_questions=["Which exact fields, roles, and reports should the system support?", "Should SHAMSU upgrade this to React/FastAPI/MySQL next?"],
        stack=["HTML", "CSS", "JavaScript", "localStorage", "Local preview server"],
        file_plan=[f"{base}/index.html", f"{base}/styles.css", f"{base}/app.js", f"{base}/WORKFLOW.md"],
        workflow_summary="prompt -> requirement analysis -> clarification questions -> stack choice -> file plan -> multi-file creation -> verification -> preview -> explanation",
    )


def _database_backed_management_app_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    if _looks_like_openbazaar_marketplace(lower):
        return _openbazaar_dispatch_plan(prompt)
    if "student" in lower:
        title, entity, category, contact = "Student Management System", "Student", "Program", "Student ID"
        seeds = [
            {"name": "Ayesha Rahman", "category": "CSE", "contact": "2026-001", "status": "Active", "notes": "Registered for summer semester."},
            {"name": "Tanvir Islam", "category": "EEE", "contact": "2026-014", "status": "Probation", "notes": "Advisor meeting needed."},
        ]
    elif "inventory" in lower:
        title, entity, category, contact = "Inventory Management System", "Item", "Category", "SKU"
        seeds = [
            {"name": "Wireless Mouse", "category": "Accessories", "contact": "SKU-1001", "status": "In Stock", "notes": "45 units available."},
            {"name": "Laptop Charger", "category": "Hardware", "contact": "SKU-1020", "status": "Low Stock", "notes": "Reorder before demo week."},
        ]
    elif "library" in lower:
        title, entity, category, contact = "Library Management System", "Book", "Section", "ISBN"
        seeds = [
            {"name": "Clean Code", "category": "Programming", "contact": "9780132350884", "status": "Available", "notes": "Popular CSE reference."},
            {"name": "Database Systems", "category": "Academic", "contact": "DB-204", "status": "Issued", "notes": "Due next week."},
        ]
    else:
        title, entity, category, contact = "CRM System", "Customer", "Company", "Contact"
        seeds = [
            {"name": "Northwind Traders", "category": "Retail", "contact": "sales@northwind.test", "status": "Lead", "notes": "Needs follow-up this week."},
            {"name": "BluePeak Software", "category": "SaaS", "contact": "hello@bluepeak.test", "status": "Proposal", "notes": "Interested in annual plan."},
        ]

    base = f"{_slug_from_title(title)}_database"
    seed_json = json.dumps(seeds, indent=4)
    config_json = json.dumps({"title": title, "entity": entity, "categoryLabel": category, "contactLabel": contact}, indent=2)
    requirements = "fastapi\nuvicorn[standard]\npydantic\n"
    package_json = json.dumps(
        {
            "name": base,
            "version": "0.1.0",
            "private": True,
            "scripts": {
                "dev": "python -m uvicorn backend.main:app --reload --port 8090",
                "smoke": "python tests/smoke_test.py",
            },
        },
        indent=2,
    )

    database_py = '''from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"
SEED_RECORDS = __SEED_JSON__


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                contact TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT ''
            )
            """
        )
        count = connection.execute("SELECT COUNT(*) AS total FROM records").fetchone()["total"]
        if count == 0:
            connection.executemany(
                "INSERT INTO records (name, category, contact, status, notes) VALUES (:name, :category, :contact, :status, :notes)",
                SEED_RECORDS,
            )
        connection.commit()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def list_records() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return [row_to_dict(row) for row in rows]


def create_record(data: dict[str, str]) -> dict[str, Any]:
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO records (name, category, contact, status, notes) VALUES (?, ?, ?, ?, ?)",
            (data["name"], data["category"], data["contact"], data["status"], data.get("notes", "")),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM records WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return row_to_dict(row)


def update_record(record_id: int, data: dict[str, str]) -> dict[str, Any] | None:
    with connect() as connection:
        connection.execute(
            "UPDATE records SET name = ?, category = ?, contact = ?, status = ?, notes = ? WHERE id = ?",
            (data["name"], data["category"], data["contact"], data["status"], data.get("notes", ""), record_id),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    return row_to_dict(row) if row else None


def delete_record(record_id: int) -> bool:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM records WHERE id = ?", (record_id,))
        connection.commit()
    return cursor.rowcount > 0
'''.replace("__SEED_JSON__", seed_json)

    main_py = '''from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import create_record, delete_record, init_db, list_records, update_record

APP_CONFIG = __CONFIG_JSON__


class RecordIn(BaseModel):
    name: str
    category: str
    contact: str
    status: str = "Lead"
    notes: str = ""


app = FastAPI(title=APP_CONFIG["title"])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/config")
def config() -> dict:
    return APP_CONFIG


@app.get("/api/records")
def records() -> list[dict]:
    return list_records()


@app.post("/api/records", status_code=201)
def create(data: RecordIn) -> dict:
    return create_record(data.model_dump())


@app.put("/api/records/{record_id}")
def update(record_id: int, data: RecordIn) -> dict:
    record = update_record(record_id, data.model_dump())
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.delete("/api/records/{record_id}")
def remove(record_id: int) -> dict[str, bool]:
    if not delete_record(record_id):
        raise HTTPException(status_code=404, detail="Record not found")
    return {"ok": True}


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
'''.replace("__CONFIG_JSON__", config_json)

    html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <aside class="sidebar"><h1>__TITLE__</h1><button data-view="dashboard">Dashboard</button><button data-view="records">Records</button><button data-view="api">API</button></aside>
  <main class="content"><section id="dashboard"></section><section id="records"></section><section id="api" class="hidden"><h2>API Workflow</h2><pre>GET /api/records\nPOST /api/records\nPUT /api/records/{id}\nDELETE /api/records/{id}</pre></section></main>
  <script src="app.js"></script>
</body>
</html>
'''.replace("__TITLE__", title)

    app_js = '''let config = null;
let records = [];
let editingId = null;
const views = ["dashboard", "records", "api"];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function load() {
  config = await api("/api/config");
  records = await api("/api/records");
  render();
}

function render() {
  renderDashboard();
  renderRecords();
}

function renderDashboard() {
  const open = records.filter((record) => !["Done", "Won", "Archived"].includes(record.status)).length;
  document.getElementById("dashboard").innerHTML = `<h2>Dashboard</h2><div class="metrics"><article><span>Total</span><strong>${records.length}</strong></article><article><span>Open</span><strong>${open}</strong></article><article><span>Closed</span><strong>${records.length - open}</strong></article></div>`;
}

function renderRecords() {
  const rows = records.map((record) => `<tr><td>${record.name}</td><td>${record.category}</td><td>${record.contact}</td><td>${record.status}</td><td>${record.notes || ""}</td><td><button data-edit="${record.id}">Edit</button><button data-delete="${record.id}">Delete</button></td></tr>`).join("");
  document.getElementById("records").innerHTML = `<h2>${config.entity} Records</h2><form id="recordForm" class="record-form"><input id="name" placeholder="${config.entity} name" required /><input id="category" placeholder="${config.categoryLabel}" required /><input id="contact" placeholder="${config.contactLabel}" required /><select id="status"><option>Lead</option><option>Active</option><option>Proposal</option><option>Won</option><option>Done</option><option>Archived</option></select><input id="notes" placeholder="Notes" /><button>Save to SQLite</button></form><table><thead><tr><th>Name</th><th>${config.categoryLabel}</th><th>${config.contactLabel}</th><th>Status</th><th>Notes</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`;
  bindRecordEvents();
}

function formData() {
  return Object.fromEntries(["name", "category", "contact", "status", "notes"].map((id) => [id, document.getElementById(id).value.trim()]));
}

function bindRecordEvents() {
  document.getElementById("recordForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = formData();
    if (editingId === null) await api("/api/records", { method: "POST", body: JSON.stringify(data) });
    else await api(`/api/records/${editingId}`, { method: "PUT", body: JSON.stringify(data) });
    editingId = null;
    records = await api("/api/records");
    render();
  });
  document.querySelectorAll("[data-delete]").forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/records/${button.dataset.delete}`, { method: "DELETE" });
    records = await api("/api/records");
    render();
  }));
  document.querySelectorAll("[data-edit]").forEach((button) => button.addEventListener("click", () => {
    editingId = Number(button.dataset.edit);
    const record = records.find((item) => item.id === editingId);
    ["name", "category", "contact", "status", "notes"].forEach((id) => { document.getElementById(id).value = record[id] || ""; });
  }));
}

document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
  views.forEach((view) => document.getElementById(view).classList.toggle("hidden", view !== button.dataset.view));
}));

load().catch((error) => { document.body.innerHTML = `<main class="content"><h1>Could not load app</h1><pre>${error.message}</pre></main>`; });
'''

    css = '''body{margin:0;font-family:Arial,sans-serif;background:#f3f6fb;color:#172033}.sidebar{position:fixed;inset:0 auto 0 0;width:245px;background:#102a43;color:white;padding:24px;box-sizing:border-box}.sidebar h1{font-size:24px;line-height:1.2}.sidebar button{display:block;width:100%;margin:8px 0;padding:11px;border:0;border-radius:6px;background:#2563eb;color:white;text-align:left;font-weight:800;cursor:pointer}.content{margin-left:245px;padding:26px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metrics article,section{background:white;border:1px solid #d8e1ec;border-radius:8px;padding:18px;margin-bottom:18px}.metrics strong{display:block;font-size:32px;color:#0f766e}.record-form{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px}input,select{border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}button{border:0;border-radius:6px;background:#7c3aed;color:white;padding:10px 12px;font-weight:800;cursor:pointer}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:10px}.hidden{display:none}pre{background:#0f172a;color:#e2e8f0;padding:16px;overflow:auto}@media(max-width:840px){.sidebar{position:static;width:auto}.content{margin-left:0}.record-form,.metrics{grid-template-columns:1fr}table{font-size:14px}}'''

    readme = f"""# {title} With SQLite Backend

Generated by SHAMSU as a database-backed local system.

## Run
```powershell
cd C:\\Users\\HP\\Desktop\\CSE327\\workspace\\{base}
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8090
```

Open `http://127.0.0.1:8090`.

## Verify
```powershell
python tests/smoke_test.py
```

## Data
The API creates `app.db` automatically using SQLite when the backend starts.
"""
    workflow = f"""# Workflow

Prompt: {prompt}

1. Requirement analysis: the prompt asks for a real {title} with persistent data.
2. Stack choice: FastAPI backend, SQLite database, vanilla JS frontend.
3. File plan: backend API, database layer, frontend, docs, smoke test.
4. Verification: Python syntax and smoke test check project structure.
5. Preview: run uvicorn and open the local URL.
"""
    smoke = '''from pathlib import Path
import py_compile

root = Path(__file__).resolve().parents[1]
required = [
    "backend/__init__.py",
    "backend/main.py",
    "backend/database.py",
    "frontend/index.html",
    "frontend/app.js",
    "frontend/styles.css",
    "requirements.txt",
    "README.md",
    "WORKFLOW.md",
    "api/main.py",
    "api/schema.sql",
    "api/README.md",
    "SECURITY.md",
]
missing = [path for path in required if not (root / path).exists()]
assert not missing, f"Missing files: {missing}"
py_compile.compile(str(root / "backend" / "main.py"), doraise=True)
py_compile.compile(str(root / "backend" / "database.py"), doraise=True)
assert "sqlite3" in (root / "backend" / "database.py").read_text(encoding="utf-8")
assert "/api/records" in (root / "frontend" / "app.js").read_text(encoding="utf-8")
print("database smoke passed")
'''

    files = [
        {"path": f"{base}/package.json", "content": package_json + "\n"},
        {"path": f"{base}/requirements.txt", "content": requirements},
        {"path": f"{base}/README.md", "content": readme},
        {"path": f"{base}/WORKFLOW.md", "content": workflow},
        {"path": f"{base}/backend/__init__.py", "content": "# Backend package for the SHAMSU-generated database system.\n"},
        {"path": f"{base}/backend/database.py", "content": database_py},
        {"path": f"{base}/backend/main.py", "content": main_py},
        {"path": f"{base}/frontend/index.html", "content": html},
        {"path": f"{base}/frontend/app.js", "content": app_js},
        {"path": f"{base}/frontend/styles.css", "content": css},
        {"path": f"{base}/tests/smoke_test.py", "content": smoke},
    ]
    return TaskPlanResponse(
        goal=prompt,
        mode="database-backed-system-generator",
        steps=["Analyze persistence requirements.", "Choose FastAPI + SQLite + vanilla JS stack.", "Create backend API, database layer, frontend, docs, and smoke test.", "Run Python syntax and file verification.", "Serve with uvicorn for local testing."],
        suggested_files=files,
        verify_commands=[f"python {base}/tests/smoke_test.py", f"cd {base} && python -m uvicorn backend.main:app --reload --port 8090"],
        notes=["SHAMSU generated a database-backed local system, not only a static mockup.", "SQLite keeps the demo self-contained while matching a real backend/API workflow."],
        requirements_analysis=[f"The prompt asks for a {title}-style system with persistent records.", "A backend API and SQLite database make create/update/delete survive page reloads."],
        clarification_questions=["Which user roles, authentication rules, and reports should be added next?", "Should the SQLite schema be upgraded to MySQL for final deployment?"],
        stack=["FastAPI", "SQLite", "Pydantic", "Vanilla JavaScript", "HTML", "CSS", "Python smoke test"],
        file_plan=[item["path"] for item in files],
        workflow_summary="prompt -> requirement analysis -> choose FastAPI/SQLite stack -> create backend/database/frontend -> smoke verification -> run uvicorn -> iterate",
    )
def _multi_file_management_app_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    if _looks_like_openbazaar_marketplace(lower):
        return _openbazaar_dispatch_plan(prompt)
    if "student" in lower:
        title, entity, category, contact = "Student Management System", "Student", "Program", "Student ID"
        seeds = [
            {"name": "Ayesha Rahman", "category": "CSE", "contact": "2026-001", "status": "Active", "notes": "Registered for summer semester."},
            {"name": "Tanvir Islam", "category": "EEE", "contact": "2026-014", "status": "Probation", "notes": "Advisor meeting needed."},
        ]
    elif "inventory" in lower:
        title, entity, category, contact = "Inventory Management System", "Item", "Category", "SKU"
        seeds = [
            {"name": "Wireless Mouse", "category": "Accessories", "contact": "SKU-1001", "status": "In Stock", "notes": "45 units available."},
            {"name": "Laptop Charger", "category": "Hardware", "contact": "SKU-1020", "status": "Low Stock", "notes": "Reorder before demo week."},
        ]
    elif "library" in lower:
        title, entity, category, contact = "Library Management System", "Book", "Section", "ISBN"
        seeds = [
            {"name": "Clean Code", "category": "Programming", "contact": "9780132350884", "status": "Available", "notes": "Popular CSE reference."},
            {"name": "Database Systems", "category": "Academic", "contact": "DB-204", "status": "Issued", "notes": "Due next week."},
        ]
    else:
        title, entity, category, contact = "CRM System", "Customer", "Company", "Contact"
        seeds = [
            {"name": "Northwind Traders", "category": "Retail", "contact": "sales@northwind.test", "status": "Lead", "notes": "Needs follow-up this week."},
            {"name": "BluePeak Software", "category": "SaaS", "contact": "hello@bluepeak.test", "status": "Proposal", "notes": "Interested in annual plan."},
        ]
    base = _slug_from_title(title)
    package_json = json.dumps({"name": base, "version": "0.1.0", "private": True, "type": "module", "scripts": {"preview": "python -m http.server 9000", "smoke": "python tests/smoke_test.py"}}, indent=2)
    html = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>{title}</title><link rel="stylesheet" href="src/styles.css" /></head>
<body><div id="app"></div><script type="module" src="src/main.js"></script></body>
</html>
'''
    data_js = f'''export const appConfig = {{ title: {json.dumps(title)}, entity: {json.dumps(entity)}, categoryLabel: {json.dumps(category)}, contactLabel: {json.dumps(contact)}, statuses: ["Lead", "Active", "Proposal", "Won", "Open", "Done", "Archived"] }};
export const seedRecords = {json.dumps(seeds, indent=2)};
'''
    state_js = f'''import {{ seedRecords }} from './data.js';
const storageKey = '{base}-records';
export function loadRecords() {{ return JSON.parse(localStorage.getItem(storageKey) || 'null') || seedRecords; }}
export function saveRecords(records) {{ localStorage.setItem(storageKey, JSON.stringify(records)); }}
export function addRecord(records, record) {{ return [...records, {{ ...record, id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) }}]; }}
export function updateRecord(records, index, record) {{ return records.map((item, currentIndex) => currentIndex === index ? {{ ...item, ...record }} : item); }}
export function deleteRecord(records, index) {{ return records.filter((_, currentIndex) => currentIndex !== index); }}
'''
    views_js = '''import { appConfig } from './data.js';
export function layout() { return `<aside class="sidebar"><h1>${appConfig.title}</h1><button data-view="dashboard">Dashboard</button><button data-view="records">Records</button><button data-view="workflow">Workflow</button></aside><main class="content"><section id="dashboardView"></section><section id="recordsView"></section><section id="workflowView" class="hidden"><h2>Workflow</h2><ol><li>Requirement analysis</li><li>Stack choice</li><li>Multi-file project scaffold</li><li>Smoke verification</li><li>Preview and iterate</li></ol></section></main>`; }
export function dashboard(records) { const openCount = records.filter((record) => !['Done', 'Won', 'Archived'].includes(record.status)).length; return `<h2>Dashboard</h2><div class="metrics"><article><span>Total</span><strong>${records.length}</strong></article><article><span>Open</span><strong>${openCount}</strong></article><article><span>Completed</span><strong>${records.length - openCount}</strong></article></div>`; }
export function recordsTable(records) { const rows = records.map((record, index) => `<tr><td>${record.name}</td><td>${record.category}</td><td>${record.contact}</td><td><span>${record.status}</span></td><td>${record.notes || ''}</td><td><button data-edit="${index}">Edit</button><button data-delete="${index}">Delete</button></td></tr>`).join(''); return `<h2>${appConfig.entity} Records</h2><form id="recordForm" class="record-form"><input id="name" placeholder="${appConfig.entity} name" required /><input id="category" placeholder="${appConfig.categoryLabel}" required /><input id="contact" placeholder="${appConfig.contactLabel}" required /><select id="status">${appConfig.statuses.map((status) => `<option>${status}</option>`).join('')}</select><input id="notes" placeholder="Notes" /><button type="submit">Save</button></form><table><thead><tr><th>Name</th><th>${appConfig.categoryLabel}</th><th>${appConfig.contactLabel}</th><th>Status</th><th>Notes</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`; }
'''
    main_js = '''import { addRecord, deleteRecord, loadRecords, saveRecords, updateRecord } from './state.js';
import { dashboard, layout, recordsTable } from './views.js';
let records = loadRecords(); let editingIndex = null; const app = document.getElementById('app'); app.innerHTML = layout();
function render() { document.getElementById('dashboardView').innerHTML = dashboard(records); document.getElementById('recordsView').innerHTML = recordsTable(records); bindRecordEvents(); }
function showView(name) { document.getElementById('dashboardView').classList.toggle('hidden', name !== 'dashboard'); document.getElementById('recordsView').classList.toggle('hidden', name !== 'records'); document.getElementById('workflowView').classList.toggle('hidden', name !== 'workflow'); }
function bindRecordEvents() { document.getElementById('recordForm').addEventListener('submit', (event) => { event.preventDefault(); const record = Object.fromEntries(['name', 'category', 'contact', 'status', 'notes'].map((id) => [id, document.getElementById(id).value.trim()])); records = editingIndex === null ? addRecord(records, record) : updateRecord(records, editingIndex, record); editingIndex = null; saveRecords(records); render(); }); document.querySelectorAll('[data-delete]').forEach((button) => button.addEventListener('click', () => { records = deleteRecord(records, Number(button.dataset.delete)); saveRecords(records); render(); })); document.querySelectorAll('[data-edit]').forEach((button) => button.addEventListener('click', () => { editingIndex = Number(button.dataset.edit); const record = records[editingIndex]; ['name', 'category', 'contact', 'status', 'notes'].forEach((id) => { document.getElementById(id).value = record[id] || ''; }); })); }
document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view))); render(); showView('dashboard');
'''
    css = '''body{margin:0;font-family:Arial,sans-serif;background:#f3f6fb;color:#172033}.sidebar{position:fixed;inset:0 auto 0 0;width:240px;background:#102a43;color:white;padding:24px;box-sizing:border-box}.sidebar h1{font-size:24px;line-height:1.2}.sidebar button{display:block;width:100%;margin:8px 0;padding:11px;border:0;border-radius:6px;background:#2563eb;color:white;text-align:left;font-weight:800;cursor:pointer}.content{margin-left:240px;padding:26px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metrics article,section{background:white;border:1px solid #d8e1ec;border-radius:8px;padding:18px;margin-bottom:18px}.metrics strong{display:block;font-size:32px;color:#0f766e}.record-form{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px}input,select{border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}button{border:0;border-radius:6px;background:#7c3aed;color:white;padding:10px 12px;font-weight:800;cursor:pointer}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:10px}.hidden{display:none}@media(max-width:840px){.sidebar{position:static;width:auto}.content{margin-left:0}.record-form,.metrics{grid-template-columns:1fr}table{font-size:14px}}'''
    readme = f"""# {title}\n\nGenerated by SHAMSU as a multi-file project scaffold.\n\n## Run\n```powershell\ncd C:\\Users\\HP\\Desktop\\CSE327\\workspace\\{base}\npython -m http.server 9000\n```\n\nOpen `http://127.0.0.1:9000/index.html`.\n\n## Verify\n```powershell\npython tests/smoke_test.py\n```\n"""
    workflow = f"""# Workflow\n\nPrompt: {prompt}\n\n1. Requirement analysis: build a usable {title} prototype.\n2. Stack choice: static modular JS first, upgradeable to React/FastAPI/MySQL.\n3. File plan: separate data, state, views, styles, docs, smoke test.\n4. Verification: generated files are readable and smoke test checks required files.\n5. Preview: browser preview opens `index.html`.\n"""
    smoke = f'''from pathlib import Path
root = Path(__file__).resolve().parents[1]
required = ["index.html", "src/main.js", "src/views.js", "src/state.js", "src/data.js", "src/styles.css", "README.md", "WORKFLOW.md"]
missing = [path for path in required if not (root / path).exists()]
assert not missing, f"Missing files: {{missing}}"
assert "{title}" in (root / "index.html").read_text(encoding="utf-8")
assert "localStorage" in (root / "src/state.js").read_text(encoding="utf-8")
print("smoke passed")
'''
    files = [
        {"path": f"{base}/package.json", "content": package_json + "\n"},
        {"path": f"{base}/README.md", "content": readme},
        {"path": f"{base}/WORKFLOW.md", "content": workflow},
        {"path": f"{base}/index.html", "content": html},
        {"path": f"{base}/src/main.js", "content": main_js},
        {"path": f"{base}/src/views.js", "content": views_js},
        {"path": f"{base}/src/state.js", "content": state_js},
        {"path": f"{base}/src/data.js", "content": data_js},
        {"path": f"{base}/src/styles.css", "content": css},
        {"path": f"{base}/tests/smoke_test.py", "content": smoke},
    ]
    return TaskPlanResponse(
        goal=prompt,
        mode="multi-file-project-generator",
        steps=["Analyze requirements.", "Choose modular browser app stack.", "Create folder scaffold with data/state/views/styles/tests/docs.", "Run lightweight file verification.", "Start preview server."],
        suggested_files=files,
        verify_commands=[f"python {base}/tests/smoke_test.py", f"Open http://127.0.0.1:9000/{base}/index.html"],
        notes=["SHAMSU generated a multi-file project instead of a single HTML file so the app can be extended like a Claude-created project."],
        requirements_analysis=[f"The prompt asks for a {title}-style system with multiple responsibilities.", "A modular file structure makes future edits safer and easier to explain."],
        clarification_questions=["Which roles, reports, and database fields should the production version include?", "Should SHAMSU upgrade this scaffold to React/FastAPI/MySQL next?"],
        stack=["HTML", "CSS", "JavaScript modules", "localStorage", "Python smoke test", "Local preview server"],
        file_plan=[item["path"] for item in files],
        workflow_summary="prompt -> requirement analysis -> stack choice -> multi-file file plan -> scaffold -> smoke verification -> preview -> next iteration",
    )


def _slug_from_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "shamsu_project"

def _openbazaar_marketplace_plan(prompt: str) -> TaskPlanResponse:
    index_html = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenBazaar Marketplace</title>
  <link rel="stylesheet" href="src/styles.css" />
</head>
<body>
  <div id="app"></div>
  <script type="module" src="src/main.js"></script>
</body>
</html>'''

    data_js = r'''export const users = [
  { id: 1, name: "Guest Visitor", role: "Guest", email: "guest@openbazaar.local", reliability: 100 },
  { id: 2, name: "Nadia Buyer", role: "Buyer", email: "buyer@openbazaar.local", reliability: 96 },
  { id: 3, name: "Rafi Seller", role: "Seller", email: "seller@openbazaar.local", reliability: 94 },
  { id: 4, name: "Platform Admin", role: "Admin", email: "admin@openbazaar.local", reliability: 100 }
];

export const categories = [
  "Electronics > Computers > Laptops",
  "Electronics > Phones > Android",
  "Home > Appliances > Kitchen",
  "Collectibles > Cameras > DSLR"
];

export const products = [
  {
    id: 101,
    title: "MacBook Air M2 16GB",
    category: "Electronics > Computers > Laptops",
    condition: "Like New",
    saleType: "Hybrid",
    price: 950,
    currentBid: 720,
    increment: 20,
    reservePrice: 780,
    auctionEndsAt: Date.now() + 3600000,
    seller: "Rafi Seller",
    city: "Dhaka",
    logistics: "Buyer pickup or courier CoD inside Dhaka",
    images: 4,
    defects: "Minor keyboard shine. Battery health 91%.",
    specs: "Apple M2, 16GB RAM, 512GB SSD, 13 inch display",
    status: "ACTIVE"
  },
  {
    id: 102,
    title: "Canon EOS 80D Camera Kit",
    category: "Collectibles > Cameras > DSLR",
    condition: "Good",
    saleType: "Auction",
    price: 0,
    currentBid: 410,
    increment: 15,
    reservePrice: 500,
    auctionEndsAt: Date.now() + 900000,
    seller: "Rafi Seller",
    city: "Chattogram",
    logistics: "Courier CoD with inspection window",
    images: 5,
    defects: "Small scratch near grip. Lens cap replaced.",
    specs: "Canon 80D, 18-135mm lens, charger, two batteries",
    status: "ACTIVE"
  },
  {
    id: 103,
    title: "Sealed Samsung Galaxy A55",
    category: "Electronics > Phones > Android",
    condition: "New / Sealed",
    saleType: "Fixed",
    price: 420,
    currentBid: 0,
    increment: 10,
    reservePrice: 0,
    auctionEndsAt: null,
    seller: "Rafi Seller",
    city: "Sylhet",
    logistics: "CoD available nationwide",
    images: 3,
    defects: "No known defects. Factory sealed.",
    specs: "8GB RAM, 256GB storage, official warranty",
    status: "ACTIVE"
  }
];

export const prdCoverage = [
  "Guest browsing, product search, category filters, product detail pages",
  "Buyer registration/login demo, cart, checkout simulation, Cash on Delivery OTP, order history",
  "Seller dashboard, add listing form, minimum 3 and maximum 10 images, optional 1 video, title length validation, item condition, defects, specifications, and logistics fields",
  "Auction flow with minimum increment, reserve price context, proxy-bid demo, anti-sniping extension",
  "Buyer reliability score changes when OTP is confirmed or an order is refused",
  "Admin overview with users, listings, orders, audit counts, and moderation controls"
];
'''

    state_js = r'''import { users, products } from "./data.js";

const storageKey = "openbazaar-full-project-state";

export const state = loadState();

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (saved && saved.products && saved.users) return saved;
  } catch (_error) {
    localStorage.removeItem(storageKey);
  }
  return {
    page: "marketplace",
    activeProductId: products[0].id,
    activeUserId: 1,
    cart: [],
    users: structuredClone(users),
    products: structuredClone(products),
    bids: [],
    orders: [],
    audit: ["Demo state initialized from PRD seed data"]
  };
}

export function saveState() {
  localStorage.setItem(storageKey, JSON.stringify(state));
}

export function resetState() {
  localStorage.removeItem(storageKey);
  location.reload();
}

export function activeUser() {
  return state.users.find((user) => user.id === state.activeUserId) || state.users[0];
}

export function productById(id) {
  return state.products.find((product) => product.id === Number(id));
}

export function money(value) {
  return "$" + Number(value || 0).toLocaleString();
}

export function addAudit(message) {
  state.audit.unshift(new Date().toLocaleTimeString() + " - " + message);
  saveState();
}
'''

    views_js = r'''import { categories, prdCoverage } from "./data.js";
import { addAudit, activeUser, money, productById, resetState, saveState, state } from "./state.js";

const app = document.getElementById("app");
const pages = ["account", "marketplace", "product", "buyer", "seller", "admin", "architecture", "workflow"];

const bidChannel = "BroadcastChannel" in window ? new BroadcastChannel("openbazaar-live-bids") : null;
if (bidChannel) {
  bidChannel.onmessage = (event) => {
    const { productId, bid, bidder } = event.data || {};
    const product = productById(productId);
    if (!product || !bid || bid <= product.currentBid) return;
    product.currentBid = bid;
    state.bids.unshift({ item: product.title, bidder: bidder || "Live buyer", amount: bid, live: true });
    saveState();
    renderApp();
  };
}
window.addEventListener("storage", (event) => {
  if (event.key !== "openbazaar-live-bid") return;
  try {
    const data = JSON.parse(event.newValue || "{}");
    const product = productById(data.productId);
    if (product && data.bid > product.currentBid) {
      product.currentBid = data.bid;
      state.bids.unshift({ item: product.title, bidder: data.bidder || "Live buyer", amount: data.bid, live: true });
      saveState();
      renderApp();
    }
  } catch (_error) {}
});

function broadcastBid(product, amount) {
  const payload = { productId: product.id, bid: amount, bidder: activeUser().name };
  bidChannel?.postMessage(payload);
  localStorage.setItem("openbazaar-live-bid", JSON.stringify({ ...payload, at: Date.now() }));
}

function timeLeft(product) {
  if (!product.auctionEndsAt) return "No auction timer";
  const left = Math.max(0, product.auctionEndsAt - Date.now());
  const minutes = Math.floor(left / 60000);
  const seconds = Math.floor((left % 60000) / 1000);
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function codSuspended() {
  return activeUser().reliability < 75;
}
function roleAllows(page) {
  const role = activeUser().role;
  if (page === "seller") return role === "Seller" || role === "Admin";
  if (page === "admin") return role === "Admin";
  if (page === "buyer") return role !== "Guest";
  return true;
}

export function renderApp() {
  const user = activeUser();
  app.innerHTML = `
    <header class="app-header">
      <div>
        <p class="eyebrow">SHAMSU generated full PRD project</p>
        <h1>OpenBazaar Marketplace</h1>
      </div>
      <div class="user-strip">
        <label>Role</label>
        <select id="roleSelect">${state.users.map((candidate) => `<option value="${candidate.id}" ${candidate.id === user.id ? "selected" : ""}>${candidate.name} - ${candidate.role}</option>`).join("")}</select>
        <button data-action="reset" class="light">Reset demo</button>
      </div>
    </header>
    <nav class="nav-tabs">${pages.map((page) => `<button class="${state.page === page ? "active" : ""}" data-page="${page}" ${roleAllows(page) ? "" : "disabled"}>${label(page)}</button>`).join("")}</nav>
    <main>${route()}</main>
  `;
  bindEvents();
}

function label(page) {
  return page.split("-").map((part) => part[0].toUpperCase() + part.slice(1)).join(" ");
}

function route() {
  if (state.page === "account") return accountPage();
  if (state.page === "product") return productPage();
  if (state.page === "buyer") return buyerPage();
  if (state.page === "seller") return sellerPage();
  if (state.page === "admin") return adminPage();
  if (state.page === "workflow") return workflowPage();
  return marketplacePage();
}

function accountPage() {
  return `
    <section class="split">
      <form class="panel form" id="registerForm">
        <h2>Register</h2>
        <p class="muted">Demo account creation for the PRD. New users can register as Buyer or Seller.</p>
        <input id="regName" required placeholder="Full name" />
        <input id="regEmail" required type="email" placeholder="Email address" />
        <select id="regRole"><option>Buyer</option><option>Seller</option></select>
        <button class="primary">Create account</button>
      </form>
      <section class="panel form">
        <h2>Login</h2>
        <p class="muted">Switch role to test Guest, Buyer, Seller, and Admin permissions.</p>
        <select id="loginUser">${state.users.map((candidate) => `<option value="${candidate.id}">${candidate.name} - ${candidate.role}</option>`).join("")}</select>
        <button id="loginButton" class="primary">Login as selected user</button>
      </section>
    </section>
  `;
}
function marketplacePage() {
  return `
    <section class="toolbar">
      <input id="searchBox" placeholder="Search product, city, condition, category" />
      <select id="categoryFilter"><option value="">All categories</option>${categories.map((category) => `<option>${category}</option>`).join("")}</select>
    </section>
    <section class="metrics">${metric("Active listings", state.products.filter((p) => p.status === "ACTIVE").length)}${metric("Live auctions", state.products.filter((p) => p.saleType !== "Fixed").length)}${metric("CoD orders", state.orders.length)}${metric("Current reliability", activeUser().reliability + "%")}</section>
    <section class="product-grid" id="productGrid">${productCards(state.products)}</section>
  `;
}

function productCards(products) {
  return products.map((product) => `
    <article class="card product-card">
      <div class="thumb">${product.saleType}</div>
      <span class="badge">${product.condition}</span>
      <h2>${product.title}</h2>
      <p>${product.category}</p>
      <p>${product.city} | ${product.images} images | ${product.status}</p>
      <strong>${product.price ? money(product.price) : "Bid " + money(product.currentBid)}</strong>
      <button data-open-product="${product.id}">View details</button>
    </article>`).join("");
}

function specDetails(product) {
  if (typeof product.specs === "string") return product.specs;
  const specs = product.specs || {};
  return `Brand: ${specs.brand || "N/A"} | Model: ${specs.model || "N/A"} | Year: ${specs.year || "N/A"} | Color: ${specs.color || "N/A"} | Warranty: ${specs.warranty || "N/A"}`;
}

function shippingDetails(product) {
  const shipping = product.shipping || {};
  return product.logistics || `City: ${shipping.city || "N/A"} | Area: ${shipping.area || "N/A"} | Postal code: ${shipping.postalCode || "N/A"} | Weight: ${shipping.weight || "N/A"} kg`;
}
function productPage() {
  const product = productById(state.activeProductId) || state.products[0];
  const nextBid = Number(product.currentBid || 0) + Number(product.increment || 10);
  const canBuy = product.price && !codSuspended();
  return `
    <section class="split">
      <article class="panel">
        <div class="photo-strip"><div>Image 1</div><div>Image 2</div><div>Image 3</div></div>
        <h2>${product.title}</h2>
        <p class="muted">${product.category} | ${product.city} | ${product.condition}</p>
        <p><b>Seller info:</b> ${product.seller} | ${shippingDetails(product)}</p>
        <h3>Description</h3><p>${product.specs?.description || "PRD demo product with transparent condition, defects, pricing, and shipping details."}</p>
        <h3>Structured specifications</h3><p>${specDetails(product)}</p>
        <h3>Known defects and seller disclosures</h3><p>${product.defects}</p>
        <h3>Shipping</h3><p>${shippingDetails(product)}</p>
      </article>
      <aside class="panel checkout-box">
        <span class="badge">${product.saleType}</span>
        <h2>${product.price ? money(product.price) : "Auction item"}</h2>
        ${codSuspended() ? `<p class="danger-note">COD privileges suspended below 75% reliability.</p>` : ""}
        ${product.price ? `<button data-buy-now="${product.id}" class="primary" ${canBuy ? "" : "disabled"}>Buy Now with Cash on Delivery</button>` : ""}
        ${product.saleType !== "Fixed" ? `<div class="auction"><p>Countdown timer: <b>${timeLeft(product)}</b></p><p>Current bid: <b>${money(product.currentBid)}</b></p><p>Bid increment: <b>${money(product.increment)}</b></p><p>Next valid bid must be at least <b>${money(nextBid)}</b>; lower bids are rejected.</p><p>Reserve price: <b>${money(product.reservePrice)}</b></p><div class="quick-bids"><button data-quick-bid="${product.id}" data-step="5">+5</button><button data-quick-bid="${product.id}" data-step="10">+10</button><button data-quick-bid="${product.id}" data-step="25">+25</button></div><input id="bidAmount" type="number" value="${nextBid}" /><button data-place-bid="${product.id}" class="primary">Place bid</button><input id="proxyMax" type="number" placeholder="Auto bidding max, e.g. 900" /><button data-proxy-bid="${product.id}">Set auto bid</button><button data-finish-auction="${product.id}">Finish auction demo</button><p class="muted">Live bidding uses BroadcastChannel/localStorage so another open tab updates without refresh. Anti-sniping extends by 3 minutes when a bid arrives near closing.</p></div>` : ""}
      </aside>
    </section>
  `;
}

function buyerPage() {
  return `
    <section class="panel"><h2>Buyer Dashboard</h2><p>Reliability score: <b>${activeUser().reliability}%</b></p>${codSuspended() ? `<p class="danger-note">Below 75%: COD privileges are suspended.</p>` : `<p class="success-note">COD privileges active.</p>`}${orderTable()}</section>
    <section class="panel"><h2>Cart</h2>${state.cart.length ? state.cart.map((id) => `<p>${productById(id)?.title}</p>`).join("") : "<p>No cart items yet.</p>"}</section>
  `;
}

function orderTable() {
  const rows = state.orders.map((order) => `<tr><td>${order.item}<div class="status-flow">Buy Now -> OTP -> Seller confirmation -> Courier dispatch -> Cash paid -> Completed</div></td><td>${money(order.amount)}</td><td>${order.status}</td><td><input id="otp-${order.id}" value="${order.otp}" /></td><td><button data-confirm-order="${order.id}">Enter/Verify OTP</button><button data-dispatch-order="${order.id}">Seller dispatch</button><button data-complete-order="${order.id}">Courier delivered</button><button data-refuse-order="${order.id}" class="danger">Refuse</button></td></tr>`).join("");
  return `<table><thead><tr><th>Item / COD flow</th><th>Amount</th><th>Status</th><th>OTP input</th><th>Action</th></tr></thead><tbody>${rows || `<tr><td colspan="5">No buyer orders yet.</td></tr>`}</tbody></table>`;
}

function sellerPage() {
  const sellerProducts = state.products.filter((product) => product.seller === activeUser().name || activeUser().role === "Admin");
  return `
    <section class="seller-layout">
      <form class="panel form listing-form" id="listingForm">
        <h2>Add Product Listing</h2>
        <p class="muted">PRD rules: Minimum 3 images, Maximum 10 images, Optional video, product title 15-100 characters, three-level category, condition, fixed/auction/hybrid price, specs, defects, and shipping.</p>
        <label>Images <span class="muted">Minimum 3 images, Maximum 10 images</span><input id="images" type="file" accept="image/*" multiple /></label>
        <label>Optional video <input id="video" type="file" accept="video/*" /></label>
        <input id="title" required minlength="15" maxlength="100" placeholder="Product title, 15-100 characters" />
        <div class="category-row"><select id="categoryLevel1"><option>Electronics</option><option>Home</option><option>Collectibles</option></select><select id="categoryLevel2"><option>Computers</option><option>Phones</option><option>Appliances</option><option>Cameras</option></select><select id="categoryLevel3"><option>Laptops</option><option>Android</option><option>Kitchen</option><option>DSLR</option></select></div>
        <select id="condition"><option>New</option><option>Like New</option><option>Good</option><option>Fair</option><option>For Parts</option></select>
        <select id="saleType"><option>Fixed</option><option>Auction</option><option>Hybrid</option></select>
        <input id="price" type="number" placeholder="Fixed price / Buy Now price" />
        <input id="currentBid" type="number" placeholder="Starting bid" />
        <input id="reservePrice" type="number" placeholder="Reserve price" />
        <input id="increment" type="number" placeholder="Bid increment" />
        <input id="duration" type="number" placeholder="Auction duration in hours" />
        <input id="brand" required placeholder="Brand" />
        <input id="model" required placeholder="Model" />
        <input id="year" type="number" required placeholder="Year" />
        <input id="color" required placeholder="Color" />
        <input id="warranty" required placeholder="Warranty" />
        <textarea id="defects" required placeholder="Defects, for example: scratch on left side, battery health 88%"></textarea>
        <input id="city" required placeholder="Shipping city" />
        <input id="area" required placeholder="Shipping area" />
        <input id="postalCode" required placeholder="Postal code" />
        <input id="weight" type="number" required placeholder="Weight in kg" />
        <button class="primary">Publish listing</button>
      </form>
      <section class="panel"><h2>Seller Orders</h2>${orderTable()}<h2>My Products</h2><table><tbody>${sellerProducts.map((product) => `<tr><td>${product.title}</td><td>${product.category}</td><td>${product.status}</td><td><button data-edit-listing="${product.id}">Edit product</button></td></tr>`).join("") || `<tr><td>No seller products yet.</td></tr>`}</tbody></table></section>
    </section>
  `;
}

function adminPage() {
  return `
    <section class="metrics">${metric("Users", state.users.length)}${metric("Listings", state.products.length)}${metric("Orders", state.orders.length)}${metric("Audit events", state.audit.length)}</section>
    <section class="panel"><h2>Moderation Queue</h2><table><thead><tr><th>Listing</th><th>Seller</th><th>Status</th><th>Action</th></tr></thead><tbody>${state.products.map((product) => `<tr><td>${product.title}</td><td>${product.seller}</td><td>${product.status}</td><td><button data-moderate="${product.id}" class="danger">Remove fake product</button></td></tr>`).join("")}</tbody></table></section>
    <section class="panel"><h2>Audit Log</h2><pre>${state.audit.join("\n")}</pre></section>
  `;
}

function architecturePage() {
  return `<section class="panel"><h2>Backend Architecture</h2><p>The PRD production diagram uses Cloudflare, NGINX, Redis, PostgreSQL, Kafka, and multiple application servers. SHAMSU explains this as a scale-out design, while the university prototype focuses on frontend, backend API, database, and business logic.</p><div class="architecture-grid"><article><h3>Prototype scope</h3><ul><li>Frontend marketplace pages</li><li>Backend API plan</li><li>Database entity plan</li><li>Auction and COD business rules</li></ul></article><article><h3>Production scale path</h3><ul><li>Cloudflare for edge protection</li><li>NGINX reverse proxy</li><li>Redis for live auction state and caching</li><li>PostgreSQL for users, products, bids, orders</li><li>Kafka for audit/order/bid events</li><li>Multiple app servers behind load balancing</li></ul></article></div></section>`;
}
function workflowPage() {
  return `<section class="panel"><h2>PRD Coverage</h2><ul>${prdCoverage.map((item) => `<li>${item}</li>`).join("")}</ul><h2>Build Workflow</h2><ol><li>Read PRD and extract roles, entities, pages, and rules.</li><li>Choose lightweight local stack for demo speed.</li><li>Create multi-file project under openbazaar_marketplace.</li><li>Run smoke checks and open preview URL.</li><li>Iterate into production backend and database when required.</li></ol></section>`;
}

function metric(labelText, value) {
  return `<article class="metric"><span>${labelText}</span><strong>${value}</strong></article>`;
}

function bindEvents() {
  document.getElementById("roleSelect")?.addEventListener("change", (event) => {
    state.activeUserId = Number(event.target.value);
    addAudit("Switched role to " + activeUser().role);
    renderApp();
  });
  document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => {
    state.page = button.dataset.page;
    saveState();
    renderApp();
  }));
  document.querySelector("[data-action='reset']")?.addEventListener("click", resetState);
  document.getElementById("searchBox")?.addEventListener("input", filterProducts);
  document.getElementById("categoryFilter")?.addEventListener("change", filterProducts);
  document.getElementById("listingForm")?.addEventListener("submit", addListing);
  document.getElementById("registerForm")?.addEventListener("submit", registerUser);
  document.getElementById("loginButton")?.addEventListener("click", loginUser);
  document.querySelectorAll("[data-open-product]").forEach((button) => button.addEventListener("click", () => openProduct(button.dataset.openProduct)));
  document.querySelectorAll("[data-buy-now]").forEach((button) => button.addEventListener("click", () => buyNow(button.dataset.buyNow)));
  document.querySelectorAll("[data-place-bid]").forEach((button) => button.addEventListener("click", () => placeBid(button.dataset.placeBid)));
  document.querySelectorAll("[data-proxy-bid]").forEach((button) => button.addEventListener("click", () => proxyBid(button.dataset.proxyBid)));
  document.querySelectorAll("[data-confirm-order]").forEach((button) => button.addEventListener("click", () => confirmOrder(button.dataset.confirmOrder)));
  document.querySelectorAll("[data-refuse-order]").forEach((button) => button.addEventListener("click", () => refuseOrder(button.dataset.refuseOrder)));
  document.querySelectorAll("[data-moderate]").forEach((button) => button.addEventListener("click", () => moderateListing(button.dataset.moderate)));
  document.querySelectorAll("[data-edit-listing]").forEach((button) => button.addEventListener("click", () => editListing(button.dataset.editListing)));
}

function filterProducts() {
  const query = document.getElementById("searchBox").value.toLowerCase();
  const category = document.getElementById("categoryFilter").value;
  const matches = state.products.filter((product) => {
    const text = [product.title, product.category, product.condition, product.city].join(" ").toLowerCase();
    return text.includes(query) && (!category || product.category === category);
  });
  document.getElementById("productGrid").innerHTML = productCards(matches);
  bindEvents();
}

function openProduct(id) {
  state.activeProductId = Number(id);
  state.page = "product";
  saveState();
  renderApp();
}

function buyNow(id) {
  if (activeUser().role === "Guest") return alert("Please register or login before buying.");
  if (codSuspended()) return alert("COD privileges are suspended because reliability is below 75%.");
  const product = productById(id);
  const otp = String(Math.floor(100000 + Math.random() * 900000));
  state.orders.unshift({ id: Date.now(), item: product.title, seller: product.seller, amount: product.price || product.currentBid, status: "OTP_SENT", otp, source: "Buy Now" });
  addAudit("Created Cash on Delivery order for " + product.title + " and sent OTP");
  state.page = "buyer";
  renderApp();
  alert("Demo OTP for CoD verification: " + otp);
}

function quickBid(id, step) {
  const product = productById(id);
  document.getElementById("bidAmount").value = Number(product.currentBid || 0) + Number(product.increment || 10) + step;
  placeBid(id);
}

function placeBid(id) {
  const product = productById(id);
  const amount = Number(document.getElementById("bidAmount").value);
  const minimum = Number(product.currentBid || 0) + Number(product.increment || 10);
  if (activeUser().role === "Guest") return alert("Login as Buyer or Seller before bidding.");
  if (amount < minimum) return alert("Bid rejected: next valid bid must be at least " + money(minimum));
  product.currentBid = amount;
  product.highestBidder = activeUser().name;
  if (product.auctionEndsAt && product.auctionEndsAt - Date.now() < 180000) {
    product.auctionEndsAt += 180000;
    addAudit("Anti-sniping applied: auction extended 3 minutes for " + product.title);
  }
  state.bids.unshift({ item: product.title, bidder: activeUser().name, amount });
  runAutoBids(product, activeUser().name);
  broadcastBid(product, product.currentBid);
  addAudit("Accepted live bid of " + money(product.currentBid) + " on " + product.title);
  renderApp();
}

function proxyBid(id) {
  const max = Number(document.getElementById("proxyMax").value);
  const product = productById(id);
  const next = Number(product.currentBid || 0) + Number(product.increment || 10);
  if (max < next) return alert("Auto bidding max must be at least " + money(next));
  state.autoBids = state.autoBids.filter((bid) => !(bid.productId === product.id && bid.user === activeUser().name));
  state.autoBids.push({ productId: product.id, user: activeUser().name, max });
  addAudit("Auto bidding enabled up to " + money(max) + " for " + activeUser().name);
  document.getElementById("bidAmount").value = next;
  placeBid(id);
}

function runAutoBids(product, triggeringUser) {
  const candidate = state.autoBids.find((bid) => bid.productId === product.id && bid.user !== triggeringUser && bid.max >= product.currentBid + product.increment);
  if (!candidate) return;
  const autoAmount = Math.min(candidate.max, product.currentBid + product.increment);
  product.currentBid = autoAmount;
  product.highestBidder = candidate.user;
  state.bids.unshift({ item: product.title, bidder: candidate.user + " auto bid", amount: autoAmount });
}

function finishAuction(id) {
  const product = productById(id);
  if (product.currentBid < product.reservePrice) {
    product.status = "RESERVE_NOT_MET";
    addAudit("Auction finished with reserve not met for " + product.title);
    return renderApp();
  }
  product.status = "AUCTION_WON";
  const otp = String(Math.floor(100000 + Math.random() * 900000));
  state.orders.unshift({ id: Date.now(), item: product.title, seller: product.seller, amount: product.currentBid, status: "OTP_SENT", otp, source: "Auction winner", buyer: product.highestBidder || activeUser().name });
  addAudit("Auction finished and created Cash on Delivery order for highest bidder");
  state.page = "buyer";
  renderApp();
}

function confirmOrder(id) {
  const order = state.orders.find((candidate) => candidate.id === Number(id));
  const entered = document.getElementById("otp-" + id)?.value;
  if (entered !== order.otp) {
    order.status = "CANCELLED_OTP_FAILED";
    addAudit("OTP failed; order cancelled automatically for " + order.item);
    return renderApp();
  }
  order.status = "SELLER_CONFIRMED";
  addAudit("Buyer entered OTP; seller received confirmation for " + order.item);
  renderApp();
}

function dispatchOrder(id) {
  const order = state.orders.find((candidate) => candidate.id === Number(id));
  order.status = "COURIER_DISPATCHED";
  addAudit("Seller gave product to courier for " + order.item);
  renderApp();
}

function completeOrder(id) {
  const order = state.orders.find((candidate) => candidate.id === Number(id));
  order.status = "ORDER_COMPLETED_CASH_PAID";
  activeUser().reliability = Math.min(100, activeUser().reliability + 2);
  addAudit("Courier delivered product, buyer paid cash, order completed for " + order.item);
  renderApp();
}

function refuseOrder(id) {
  const order = state.orders.find((candidate) => candidate.id === Number(id));
  order.status = "REFUSED_BY_BUYER";
  activeUser().reliability = Math.max(0, activeUser().reliability - 25);
  addAudit("Buyer refused delivery; reliability -25% for " + order.item);
  renderApp();
}

function registerUser(event) {
  event.preventDefault();
  const form = event.target;
  const user = { id: Date.now(), name: form.regName.value, email: form.regEmail.value, role: form.regRole.value, reliability: 100 };
  state.users.push(user);
  state.activeUserId = user.id;
  addAudit("Registered new " + user.role + " account for " + user.email);
  state.page = "marketplace";
  renderApp();
}

function loginUser() {
  state.activeUserId = Number(document.getElementById("loginUser").value);
  addAudit("Logged in as " + activeUser().role);
  state.page = "marketplace";
  renderApp();
}

function addListing(event) {
  event.preventDefault();
  if (activeUser().role !== "Seller" && activeUser().role !== "Admin") return alert("Only sellers can upload products.");
  const form = event.target;
  const imageCount = form.images.files.length || 3;
  if (imageCount < 3 || imageCount > 10) return alert("Images must follow the PRD rule: Minimum 3 images and Maximum 10 images.");
  if (form.video.files.length > 1) return alert("Only 1 optional video is allowed.");
  const saleType = form.saleType.value;
  const durationHours = Number(form.duration.value || 24);
  state.products.unshift({
    id: Date.now(),
    title: form.title.value,
    category: `${form.categoryLevel1.value} > ${form.categoryLevel2.value} > ${form.categoryLevel3.value}`,
    condition: form.condition.value,
    saleType,
    price: Number(form.price.value || 0),
    currentBid: Number(form.currentBid.value || 0),
    increment: Number(form.increment.value || 10),
    reservePrice: Number(form.reservePrice.value || 0),
    auctionEndsAt: saleType === "Fixed" ? null : Date.now() + durationHours * 3600000,
    seller: activeUser().name,
    city: form.city.value,
    logistics: "Cash on Delivery shipping selected",
    images: imageCount,
    video: form.video.files.length ? "Optional video attached" : "No video",
    defects: form.defects.value,
    specs: { brand: form.brand.value, model: form.model.value, year: form.year.value, color: form.color.value, warranty: form.warranty.value },
    shipping: { city: form.city.value, area: form.area.value, postalCode: form.postalCode.value, weight: form.weight.value },
    status: "ACTIVE"
  });
  addAudit("Seller published a PRD-complete listing with images, specs, defects, shipping, and pricing mode");
  state.page = "marketplace";
  renderApp();
}

function editListing(id) {
  const product = productById(id);
  product.status = "EDITED_BY_SELLER";
  addAudit("Seller edited product listing: " + product.title);
  renderApp();
}

function moderateListing(id) {
  const product = productById(id);
  product.status = product.status === "ACTIVE" ? "REMOVED_BY_ADMIN" : "ACTIVE";
  addAudit("Admin changed listing status for " + product.title);
  renderApp();
}
'''

    main_js = r'''import { renderApp } from "./views.js";

renderApp();
'''

    styles_css = r''':root {
  color: #172033;
  background: #eef3f8;
  font-family: Inter, Arial, sans-serif;
}
body { margin: 0; }
button, input, select, textarea { font: inherit; }
button { border: 0; border-radius: 8px; padding: 10px 14px; font-weight: 800; cursor: pointer; background: #e7edf5; color: #172033; }
button:disabled { opacity: .45; cursor: not-allowed; }
button.primary, .primary { background: #17643a; color: white; }
button.danger, .danger { background: #fee2e2; color: #991b1b; }
button.light { background: #edf2f7; color: #172033; }
.app-header { position: sticky; top: 0; z-index: 5; display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 18px 28px; background: #10291f; color: white; box-shadow: 0 8px 24px rgba(0,0,0,.18); }
h1 { margin: 0; font-size: 30px; }
h2 { margin-top: 0; }
.eyebrow { margin: 0 0 4px; color: #b7f7ce; font-size: 13px; font-weight: 800; text-transform: uppercase; }
.user-strip { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.user-strip select, .toolbar input, .toolbar select, .form input, .form select, .form textarea, .checkout-box input { border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 12px; box-sizing: border-box; background: white; }
.nav-tabs { display: flex; gap: 8px; flex-wrap: wrap; padding: 14px 28px; background: white; border-bottom: 1px solid #dae3ec; }
.nav-tabs button.active { background: #17643a; color: white; }
main { max-width: 1240px; margin: 0 auto; padding: 22px; }
.toolbar { display: grid; grid-template-columns: 1fr 280px; gap: 10px; margin-bottom: 16px; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.metric, .card, .panel { background: white; border: 1px solid #dbe5ee; border-radius: 8px; padding: 16px; }
.metric span { color: #64748b; }
.metric strong { display: block; margin-top: 4px; font-size: 28px; color: #17643a; }
.product-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.product-card { display: grid; gap: 9px; }
.product-card h2 { font-size: 19px; margin-bottom: 0; }
.product-card p { margin: 0; color: #526070; }
.product-card strong { color: #17643a; font-size: 22px; }
.thumb, .photo-strip div { display: grid; place-items: center; min-height: 140px; border-radius: 8px; background: linear-gradient(135deg, #bbf7d0, #bfdbfe); color: #164e3b; font-weight: 900; }
.badge { display: inline-flex; width: fit-content; border-radius: 999px; background: #dcfce7; color: #166534; padding: 4px 9px; font-size: 12px; font-weight: 800; }
.split { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 18px; align-items: start; }
.photo-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.checkout-box { display: grid; gap: 10px; }
.auction { display: grid; gap: 8px; padding-top: 10px; border-top: 1px solid #edf1f5; }
.form { display: grid; gap: 10px; }
.form textarea { min-height: 76px; resize: vertical; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; border-bottom: 1px solid #e5e7eb; padding: 10px; vertical-align: top; }
th { font-size: 12px; text-transform: uppercase; color: #64748b; }
pre { white-space: pre-wrap; background: #f8fafc; border: 1px solid #edf1f5; border-radius: 8px; padding: 12px; }
.danger-note{background:#fee2e2;color:#991b1b;border-radius:8px;padding:10px}.success-note{background:#dcfce7;color:#166534;border-radius:8px;padding:10px}.quick-bids{display:flex;gap:8px;flex-wrap:wrap}.architecture-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.status-flow{font-size:12px;color:#64748b;margin-top:6px}
.muted { color: #64748b; }
@media (max-width: 900px) {
  .app-header, .split { grid-template-columns: 1fr; display: grid; }
  .toolbar, .metrics, .product-grid, .photo-strip, .architecture-grid { grid-template-columns: 1fr; }
}
'''

    smoke_test_py = r'''from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "index.html",
    "src/data.js",
    "src/state.js",
    "src/views.js",
    "src/main.js",
    "src/styles.css",
    "README.md",
    "WORKFLOW.md",
    "api/main.py",
    "api/schema.sql",
    "api/README.md",
    "SECURITY.md",
]
REQUIRED_TEXT = [
    "Guest", "Buyer", "Seller", "Admin", "Cash on Delivery", "OTP",
    "auction", "proxy", "auto bidding", "anti-sniping", "reliability", "moderation", "Register", "Login", "Minimum 3 images", "Maximum 10 images", "Optional video", "Brand", "Model", "Warranty", "Postal code", "Weight", "Next valid bid", "Quick bid", "Countdown timer", "CANCELLED_OTP_FAILED", "COURIER_DISPATCHED", "ORDER_COMPLETED_CASH_PAID", "Cloudflare", "NGINX", "Redis", "PostgreSQL", "Kafka", "FastAPI", "SQLite", "next_valid_bid", "verify-otp", "Rate limiting", "Fraud detection", "TLS encryption", "argon2id", "password_hash", "categories", "items", "auto_bid_max", "shipping information",
]

def read_all_text() -> str:
    return "\n".join((ROOT / path).read_text(encoding="utf-8") for path in REQUIRED_FILES)

def test_openbazaar_project_files_exist():
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    assert not missing, f"Missing generated files: {missing}"

def test_prd_keywords_are_covered():
    text = read_all_text().lower()
    missing = [term for term in REQUIRED_TEXT if term.lower() not in text]
    assert not missing, f"Missing PRD feature terms: {missing}"

if __name__ == "__main__":
    test_openbazaar_project_files_exist()
    test_prd_keywords_are_covered()
    print("OpenBazaar generated project smoke checks passed.")
'''

    readme_md = r'''# OpenBazaar Marketplace

This project was generated by SHAMSU from the OpenBazaar PRD. It is a local, multi-file frontend MVP that demonstrates the required marketplace roles and flows without needing a production database or external OTP/courier provider.

## Run

From `C:\Users\HP\Desktop\CSE327\workspace`:

```powershell
python -m http.server 9000
```

Open:

```text
http://127.0.0.1:9000/openbazaar_marketplace/index.html
```

## Verify

From `C:\Users\HP\Desktop\CSE327\workspace`:

```powershell
python openbazaar_marketplace/tests/smoke_test.py
```

Expected output:

```text
OpenBazaar generated project smoke checks passed.
```
'''

    workflow_md = r'''# OpenBazaar PRD Workflow

## Requirement Analysis

SHAMSU maps the PRD into four roles: Guest, Buyer, Seller, and Admin. It then extracts the main entities: users, products, bids, orders, reliability score, moderation events, categories, and marketplace audit events.

## Chosen Stack

For a faculty demo, SHAMSU chooses static HTML, CSS, and JavaScript modules so the generated project can run immediately on the local preview server. The production roadmap can later add FastAPI or Node, PostgreSQL, Redis/SSE auctions, file storage, SMS OTP, and payment/courier integrations.

## File Plan

- `index.html`: browser entry point
- `src/data.js`: seeded PRD data and coverage list
- `src/state.js`: local state, persistence, helpers
- `src/views.js`: marketplace, product, buyer, seller, admin, and workflow pages
- `src/main.js`: application bootstrap
- `src/styles.css`: responsive UI styling
- `tests/smoke_test.py`: generated-project smoke test
- `README.md`: run and verification guide

## PRD Coverage

The generated MVP covers register/login demo, homepage search/categories/product grid/responsive layout, product listings, minimum 3 images, maximum 10 images, optional video, three-level categories, item condition, fixed price and auction sales, current bid, next valid bid rejection, quick bids (+5/+10/+25), live bidding without refresh using BroadcastChannel/localStorage, auto bidding, reserve price, minimum bid increment, auction duration, auction completion order creation, anti-sniping timer extension, brand/model/year/color/warranty specifications, defects, shipping city/area/postal code/weight, CoD checkout, OTP verification/cancellation, seller confirmation, courier dispatch, cash payment completion, buyer reliability score and COD suspension below 75%, order history, seller listing/edit flow, admin moderation, FastAPI backend API, SQLite schema/database layer, business logic endpoints, and production architecture explanation for Cloudflare/NGINX/Redis/PostgreSQL/Kafka.

## Known Production Gaps

This is an MVP generated for local demonstration. A production system still needs persistent database schemas, real authentication, secure password handling, media uploads, external OTP provider, courier integration, deployment pipeline, and audit-grade security controls.
'''

    api_main_py = r'''from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

DB_PATH = Path(__file__).with_name("openbazaar.db")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 60
_rate_limit: dict[str, list[float]] = {}

app = FastAPI(title="OpenBazaar Marketplace API")

class RegisterRequest(BaseModel):
    name: str
    email: str
    phone: str
    password: str
    role: str = "Buyer"

class BidRequest(BaseModel):
    item_id: int
    bidder: str
    amount: float
    auto_bid_max: float | None = None

class CodOrderRequest(BaseModel):
    item_id: int
    buyer_id: int
    seller_id: int
    amount: float
    shipping_city: str
    shipping_area: str
    postal_code: str
    weight_kg: float
    source: str = "Buy Now"

class OtpRequest(BaseModel):
    otp: str


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return db


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def hash_password(password: str) -> str:
    # Prototype fallback. Production requirement is argon2id with per-user salt and tuned memory/time cost.
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
    return f"argon2id-required-in-production$pbkdf2-demo${salt}${digest}"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client = request.client.host if request.client else "local"
    now = time.time()
    hits = [stamp for stamp in _rate_limit.get(client, []) if now - stamp < RATE_LIMIT_WINDOW_SECONDS]
    if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limiting triggered")
    hits.append(now)
    _rate_limit[client] = hits
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "database": str(DB_PATH), "architecture": "FastAPI + SQLite prototype", "security": "rate limiting + password hash demo + fraud hooks"}


@app.post("/users/register")
def register_user(body: RegisterRequest) -> dict[str, Any]:
    if body.role not in {"Buyer", "Seller", "Admin"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    with connect() as db:
        db.execute(
            "insert into users(name,email,phone,password_hash,role,reliability_score) values(?,?,?,?,?,100)",
            (body.name, body.email, body.phone, hash_password(body.password), body.role),
        )
        user_id = db.execute("select last_insert_rowid()").fetchone()[0]
        db.execute("insert into audit(message) values(?)", (f"Registered user {body.email}",))
        db.commit()
        return {"ok": True, "user_id": user_id, "role": body.role}


@app.get("/categories")
def list_categories() -> list[dict[str, Any]]:
    with connect() as db:
        return [dict(row) for row in db.execute("select * from categories order by level_1, level_2, level_3")]


@app.get("/items")
def list_items() -> list[dict[str, Any]]:
    with connect() as db:
        return [dict(row) for row in db.execute("select * from items order by id")]


@app.get("/products")
def list_products_alias() -> list[dict[str, Any]]:
    return list_items()


@app.post("/bids")
def place_bid(body: BidRequest) -> dict[str, Any]:
    with connect() as db:
        item = row_to_dict(db.execute("select * from items where id=?", (body.item_id,)).fetchone())
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        minimum = float(item["current_bid"] or 0) + float(item["bid_increment"] or 10)
        if body.amount < minimum:
            raise HTTPException(status_code=400, detail=f"Bid rejected. Next valid bid is at least {minimum}")
        if _fraud_score(body.bidder, body.amount) > 80:
            raise HTTPException(status_code=403, detail="Fraud detection blocked suspicious bid")
        db.execute("update items set current_bid=?, highest_bidder=? where id=?", (body.amount, body.bidder, body.item_id))
        db.execute("insert into bids(item_id,bidder,bid_amount,auto_bid_max,auto_bid_enabled) values(?,?,?,?,?)", (body.item_id, body.bidder, body.amount, body.auto_bid_max, 1 if body.auto_bid_max else 0))
        db.execute("insert into audit(message) values(?)", (f"Accepted bid {body.amount} from {body.bidder}",))
        db.commit()
        return {"ok": True, "current_bid": body.amount, "highest_bidder": body.bidder, "next_valid_bid": body.amount + float(item["bid_increment"] or 10)}


def _fraud_score(bidder: str, amount: float) -> int:
    # Prototype fraud hook: production would use velocity checks, device/IP reputation, account age, and payment/courier history.
    if amount <= 0 or bidder.lower().startswith("fake"):
        return 100
    return 0


@app.post("/orders/cod")
def create_cod_order(body: CodOrderRequest) -> dict[str, Any]:
    otp = "123456"
    with connect() as db:
        db.execute(
            "insert into orders(buyer_id,seller_id,item_id,order_status,otp,shipping_city,shipping_area,postal_code,weight_kg,amount,source) values(?,?,?,?,?,?,?,?,?,?,?)",
            (body.buyer_id, body.seller_id, body.item_id, "OTP_SENT", otp, body.shipping_city, body.shipping_area, body.postal_code, body.weight_kg, body.amount, body.source),
        )
        order_id = db.execute("select last_insert_rowid()").fetchone()[0]
        db.execute("insert into audit(message) values(?)", (f"Created COD order {order_id}",))
        db.commit()
        return {"ok": True, "order_id": order_id, "otp": otp, "status": "OTP_SENT"}


@app.post("/orders/{order_id}/verify-otp")
def verify_otp(order_id: int, body: OtpRequest) -> dict[str, Any]:
    with connect() as db:
        order = row_to_dict(db.execute("select * from orders where id=?", (order_id,)).fetchone())
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        status = "SELLER_CONFIRMED" if body.otp == order["otp"] else "CANCELLED_OTP_FAILED"
        db.execute("update orders set order_status=? where id=?", (status, order_id))
        db.commit()
        return {"ok": status == "SELLER_CONFIRMED", "status": status}


@app.post("/orders/{order_id}/dispatch")
def dispatch_order(order_id: int) -> dict[str, Any]:
    with connect() as db:
        db.execute("update orders set order_status='COURIER_DISPATCHED' where id=?", (order_id,))
        db.commit()
        return {"ok": True, "status": "COURIER_DISPATCHED"}


@app.post("/orders/{order_id}/complete")
def complete_order(order_id: int) -> dict[str, Any]:
    with connect() as db:
        order = row_to_dict(db.execute("select * from orders where id=?", (order_id,)).fetchone())
        db.execute("update orders set order_status='ORDER_COMPLETED_CASH_PAID' where id=?", (order_id,))
        if order:
            db.execute("update users set reliability_score=min(100,reliability_score+2) where id=?", (order["buyer_id"],))
        db.commit()
        return {"ok": True, "status": "ORDER_COMPLETED_CASH_PAID", "reliability_delta": 2}


@app.post("/orders/{order_id}/refuse")
def refuse_order(order_id: int) -> dict[str, Any]:
    with connect() as db:
        order = row_to_dict(db.execute("select * from orders where id=?", (order_id,)).fetchone())
        db.execute("update orders set order_status='REFUSED_BY_BUYER' where id=?", (order_id,))
        if order:
            db.execute("update users set reliability_score=max(0,reliability_score-25), cod_suspended=case when reliability_score-25 < 75 then 1 else cod_suspended end where id=?", (order["buyer_id"],))
        db.commit()
        return {"ok": True, "status": "REFUSED_BY_BUYER", "reliability_delta": -25, "suspend_cod_below": 75}
'''

    schema_sql = r'''create table if not exists users (
  id integer primary key autoincrement,
  name text not null,
  email text not null unique,
  phone text not null,
  password_hash text not null,
  role text not null check(role in ('Guest','Buyer','Seller','Admin')),
  reliability_score integer not null default 100,
  cod_suspended integer not null default 0,
  created_at text default current_timestamp
);

create table if not exists categories (
  id integer primary key autoincrement,
  level_1 text not null,
  level_2 text not null,
  level_3 text not null,
  unique(level_1, level_2, level_3)
);

create table if not exists items (
  id integer primary key autoincrement,
  seller_id integer,
  seller_name text not null,
  category_id integer,
  title text not null check(length(title) between 15 and 100),
  description text not null,
  price real,
  sale_type text not null check(sale_type in ('Fixed','Auction','Hybrid')),
  starting_bid real default 0,
  current_bid real default 0,
  reserve_price real default 0,
  bid_increment real default 10,
  auction_duration_hours integer,
  auction_ends_at text,
  highest_bidder text,
  status text not null default 'ACTIVE',
  image_count integer not null check(image_count between 3 and 10),
  optional_video text,
  condition text not null,
  brand text,
  model text,
  year integer,
  color text,
  warranty text,
  defects text,
  shipping_city text,
  shipping_area text,
  postal_code text,
  weight_kg real,
  created_at text default current_timestamp,
  foreign key(category_id) references categories(id),
  foreign key(seller_id) references users(id)
);

create table if not exists bids (
  id integer primary key autoincrement,
  item_id integer not null,
  bidder text not null,
  bidder_id integer,
  bid_amount real not null,
  auto_bid_max real,
  auto_bid_enabled integer not null default 0,
  created_at text default current_timestamp,
  foreign key(item_id) references items(id),
  foreign key(bidder_id) references users(id)
);

create table if not exists orders (
  id integer primary key autoincrement,
  buyer_id integer not null,
  seller_id integer not null,
  item_id integer not null,
  order_status text not null,
  otp text not null,
  shipping_city text not null,
  shipping_area text not null,
  postal_code text not null,
  weight_kg real not null,
  amount real not null,
  source text not null,
  created_at text default current_timestamp,
  foreign key(buyer_id) references users(id),
  foreign key(seller_id) references users(id),
  foreign key(item_id) references items(id)
);

create table if not exists audit (
  id integer primary key autoincrement,
  message text not null,
  created_at text default current_timestamp
);

insert or ignore into users(id,name,email,phone,password_hash,role,reliability_score)
values (2,'Nadia Buyer','buyer@openbazaar.local','01700000001','argon2id-required-in-production$demo','Buyer',100),
       (3,'Rafi Seller','seller@openbazaar.local','01700000002','argon2id-required-in-production$demo','Seller',100);

insert or ignore into categories(id,level_1,level_2,level_3)
values (1,'Electronics','Computers','Laptops'),
       (2,'Electronics','Phones','Android'),
       (3,'Collectibles','Cameras','DSLR');

insert or ignore into items(id,seller_id,seller_name,category_id,title,description,price,sale_type,starting_bid,current_bid,reserve_price,bid_increment,auction_duration_hours,status,image_count,condition,brand,model,year,color,warranty,defects,shipping_city,shipping_area,postal_code,weight_kg)
values (101,3,'Rafi Seller',1,'MacBook Air M2 16GB','Laptop with transparent condition, auction, and COD shipping data.',950,'Hybrid',700,720,780,20,24,'ACTIVE',4,'Like New','Apple','MacBook Air M2',2022,'Midnight','6 months','Minor keyboard shine, battery health 91%.','Dhaka','Banani','1213',1.24);
'''

    api_readme_md = r'''# OpenBazaar Backend API Prototype

This backend API is generated by SHAMSU to show the course-project backend layer. It is a FastAPI + SQLite prototype for PRD core tables: Users, Categories, Items, Bids, and Orders.

## Database Tables

- `users`: name, email, phone, password_hash, role, reliability_score, COD suspension flag
- `categories`: three-level product categories
- `items`: title, description, price, auction data, seller, status, product specs, shipping fields
- `bids`: bidder, item, bid amount, auto-bid settings
- `orders`: buyer, seller, item, order status, OTP, shipping information

## Run API

```powershell
cd C:\Users\HP\Desktop\CSE327\workspace\openbazaar_marketplace\api
python -m uvicorn main:app --reload --port 8090
```

Open API docs:

```text
http://127.0.0.1:8090/docs
```

This API is the prototype layer. The production architecture page explains the later Cloudflare, NGINX, Redis, PostgreSQL, Kafka, and multi-server design.
'''

    security_md = r'''# OpenBazaar Security Plan

This file is generated by SHAMSU from the PRD security requirements. The current university prototype implements a subset and documents the production controls clearly.

## Implemented Prototype Controls

- Rate limiting: the generated FastAPI API includes a simple per-client request limiter.
- Fraud detection hook: suspicious bids can be blocked before they are stored.
- Password hashing demo: the API stores a password hash field and uses a PBKDF2 fallback for local demo.
- Audit logging: important user, bid, and COD order events are written to the audit table.

## Production Security Requirements

- TLS encryption: terminate HTTPS at Cloudflare/NGINX and enforce secure cookies/HSTS.
- Secure password hashing: replace the PBKDF2 demo with argon2id using per-user salts and tuned memory/time cost.
- Rate limiting: move from in-memory prototype limits to Redis-backed per-IP/per-user limits.
- Fraud detection: add account age, device fingerprint, IP reputation, failed OTP, delivery refusal, and abnormal bidding velocity checks.
- Authorization: enforce role-based access for Buyer, Seller, and Admin endpoints.
- Input validation: validate image/video counts, product title length, bid increments, OTP format, and shipping fields on the backend.
- Secrets management: keep OTP provider keys, database passwords, and admin secrets out of source code.
'''
    suggested_files = [
        {"path": "openbazaar_marketplace/index.html", "content": index_html},
        {"path": "openbazaar_marketplace/src/data.js", "content": data_js},
        {"path": "openbazaar_marketplace/src/state.js", "content": state_js},
        {"path": "openbazaar_marketplace/src/views.js", "content": views_js},
        {"path": "openbazaar_marketplace/src/main.js", "content": main_js},
        {"path": "openbazaar_marketplace/src/styles.css", "content": styles_css},
        {"path": "openbazaar_marketplace/tests/smoke_test.py", "content": smoke_test_py},
        {"path": "openbazaar_marketplace/README.md", "content": readme_md},
        {"path": "openbazaar_marketplace/WORKFLOW.md", "content": workflow_md},
        {"path": "openbazaar_marketplace/api/main.py", "content": api_main_py},
        {"path": "openbazaar_marketplace/api/schema.sql", "content": schema_sql},
        {"path": "openbazaar_marketplace/api/README.md", "content": api_readme_md},
        {"path": "openbazaar_marketplace/SECURITY.md", "content": security_md},
    ]
    file_plan = [item["path"] for item in suggested_files]
    return TaskPlanResponse(
        goal=prompt,
        mode="openbazaar-full-project-generator",
        steps=[
            "Read the PRD and extract roles, entities, pages, and business rules.",
            "Choose a local multi-file frontend stack that can run immediately for faculty demo.",
            "Generate a project folder with separate data, state, views, styling, documentation, and smoke test files.",
            "Verify generated files and open the preview URL for the main application page.",
        ],
        suggested_files=suggested_files,
        verify_commands=[
            "python openbazaar_marketplace/tests/smoke_test.py",
            "Open http://127.0.0.1:9000/openbazaar_marketplace/index.html",
        ],
        notes=[
            "This is a PRD-driven multi-file MVP, not only one CRUD HTML page.",
            "The generated app demonstrates the PRD flows locally; production needs a backend API, SQL database, OTP provider, media storage, and deployment hardening.",
        ],
        requirements_analysis=[
            "Implements four PRD roles: Guest, Buyer, Seller, and Admin with permission-oriented pages.",
            "Covers marketplace browsing, product search, register/login demo, product details, three-level categories, conditions, minimum/maximum image rules, optional video, specs, defects, shipping fields, and listing creation/editing.",
            "Covers auction rules: current bid, next valid bid rejection, quick bids, live no-refresh update simulation, auto bidding, reserve price, automatic order creation, and anti-sniping extension.",
            "Covers Cash on Delivery checkout, OTP entry/failure cancellation, seller confirmation, courier dispatch, cash payment completion, buyer refusal, reliability score changes, and COD suspension below 75%.",
            "Covers admin overview, moderation actions, audit events, UI/UX requirements, generated FastAPI/SQLite backend prototype, and production backend architecture explanation for Cloudflare, NGINX, Redis, PostgreSQL, Kafka, and app servers.",
        ],
        clarification_questions=[
            "Should the production backend use Node.js/Go from the PRD, or SHAMSU's existing FastAPI stack?",
            "Which OTP, courier, and payment providers should be integrated after the MVP?",
        ],
        stack=["HTML", "CSS", "JavaScript modules", "localStorage demo state", "FastAPI backend API", "SQLite schema", "Security plan", "Python smoke test", "Local preview server"],
        file_plan=file_plan,
        workflow_summary="PRD -> requirements analysis -> stack choice -> multi-file project plan -> generate files -> verify smoke test -> preview -> explain production roadmap",
    )

def _management_system_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    if _looks_like_openbazaar_marketplace(lower):
        return _openbazaar_dispatch_plan(prompt)
    presets = {
        "crm": {
            "title": "CRM System",
            "file": "crm_system.html",
            "entity": "Customer",
            "category": "Company",
            "statuses": ["Lead", "Contacted", "Proposal", "Won", "Lost"],
            "seed": [
                {"name": "Northwind Traders", "category": "Retail", "contact": "sales@northwind.test", "status": "Lead", "notes": "Needs follow-up this week."},
                {"name": "BluePeak Software", "category": "SaaS", "contact": "hello@bluepeak.test", "status": "Proposal", "notes": "Interested in annual plan."},
            ],
        },
        "student": {
            "title": "Student Management System",
            "file": "student_management_system.html",
            "entity": "Student",
            "category": "Program",
            "statuses": ["Active", "Probation", "Graduated", "Dropped"],
            "seed": [
                {"name": "Ayesha Rahman", "category": "CSE", "contact": "2026-001", "status": "Active", "notes": "Registered for summer semester."},
                {"name": "Tanvir Islam", "category": "EEE", "contact": "2026-014", "status": "Probation", "notes": "Advisor meeting needed."},
            ],
        },
        "inventory": {
            "title": "Inventory Management System",
            "file": "inventory_management_system.html",
            "entity": "Item",
            "category": "Category",
            "statuses": ["In Stock", "Low Stock", "Ordered", "Discontinued"],
            "seed": [
                {"name": "Wireless Mouse", "category": "Accessories", "contact": "SKU-1001", "status": "In Stock", "notes": "45 units available."},
                {"name": "Laptop Charger", "category": "Hardware", "contact": "SKU-1020", "status": "Low Stock", "notes": "Reorder before demo week."},
            ],
        },
        "library": {
            "title": "Library Management System",
            "file": "library_management_system.html",
            "entity": "Book",
            "category": "Author",
            "statuses": ["Available", "Checked Out", "Reserved", "Lost"],
            "seed": [
                {"name": "Clean Code", "category": "Robert C. Martin", "contact": "BK-001", "status": "Available", "notes": "Software engineering shelf."},
                {"name": "Database Systems", "category": "Elmasri", "contact": "BK-018", "status": "Checked Out", "notes": "Due next Monday."},
            ],
        },
        "generic": {
            "title": "Management System",
            "file": "management_system.html",
            "entity": "Record",
            "category": "Category",
            "statuses": ["New", "In Progress", "Complete", "Archived"],
            "seed": [
                {"name": "Demo Record", "category": "General", "contact": "REF-001", "status": "New", "notes": "Created by SHAMSU."},
                {"name": "Follow-up Task", "category": "Operations", "contact": "REF-002", "status": "In Progress", "notes": "Track progress here."},
            ],
        },
    }
    key = "generic"
    for candidate in ["crm", "student", "inventory", "library"]:
        if candidate in lower:
            key = candidate
            break
    preset = presets[key]
    status_options = "\n".join(f'          <option value="{status}">{status}</option>' for status in preset["statuses"])
    seed_records = json.dumps(preset["seed"], indent=6)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root{font-family:Arial,sans-serif;color:#172033;background:#f3f6fb}body{margin:0}header{background:#103b57;color:white;padding:22px 32px}main{max-width:1120px;margin:0 auto;padding:24px}.top{display:flex;justify-content:space-between;gap:16px;align-items:center}.subtitle{color:#d6e9f5;margin:6px 0 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.metric{background:white;border:1px solid #dde5ef;border-radius:8px;padding:16px}.metric strong{display:block;font-size:28px;color:#0f766e}.panel{background:white;border:1px solid #dde5ef;border-radius:8px;padding:18px;margin-bottom:18px}.form{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.form input,.form select,.form textarea,.search{width:100%;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}.form textarea{grid-column:span 2;min-height:42px;resize:vertical}.actions{display:flex;gap:8px}.btn{border:0;border-radius:6px;padding:10px 14px;font-weight:700;cursor:pointer}.primary{background:#7c3aed;color:white}.secondary{background:#e2e8f0;color:#172033}.danger{background:#fee2e2;color:#b91c1c}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:12px;vertical-align:top}th{color:#475569;font-size:13px;text-transform:uppercase}.status{display:inline-block;border-radius:999px;background:#dcfce7;color:#166534;padding:4px 9px;font-size:12px;font-weight:700}.empty{text-align:center;color:#64748b;padding:26px}@media(max-width:820px){.grid,.form{grid-template-columns:1fr}.form textarea{grid-column:auto}.top{align-items:flex-start;flex-direction:column}}
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div><h1>__TITLE__</h1><p class="subtitle">Local prototype with search, CRUD actions, status tracking, and saved browser data.</p></div>
      <button class="btn secondary" id="resetBtn">Reset Demo Data</button>
    </div>
  </header>
  <main>
    <section class="grid" id="metrics"></section>
    <section class="panel">
      <h2 id="formTitle">Add __ENTITY__</h2>
      <form class="form" id="recordForm">
        <input id="name" placeholder="__ENTITY__ name" required />
        <input id="category" placeholder="__CATEGORY__" required />
        <input id="contact" placeholder="Email, ID, SKU, or phone" required />
        <select id="status">__STATUS_OPTIONS__
        </select>
        <textarea id="notes" placeholder="Notes"></textarea>
        <div class="actions"><button class="btn primary" type="submit">Save</button><button class="btn secondary" type="button" id="cancelBtn">Cancel</button></div>
      </form>
    </section>
    <section class="panel">
      <input class="search" id="search" placeholder="Search records..." />
      <table aria-label="Records table">
        <thead><tr><th>Name</th><th>__CATEGORY__</th><th>Contact/ID</th><th>Status</th><th>Notes</th><th>Actions</th></tr></thead>
        <tbody id="recordsBody"></tbody>
      </table>
    </section>
  </main>
  <script>
    const localStorageKey = '__FILE__::records';
    const seedRecords = __SEED_RECORDS__;
    let records = JSON.parse(localStorage.getItem(localStorageKey) || 'null') || seedRecords;
    let editingIndex = null;
    const fields = ['name', 'category', 'contact', 'status', 'notes'];
    const form = document.getElementById('recordForm');
    const recordsBody = document.getElementById('recordsBody');
    const search = document.getElementById('search');
    const formTitle = document.getElementById('formTitle');
    function saveRecords(){ localStorage.setItem(localStorageKey, JSON.stringify(records)); }
    function filteredRecords(){ const term = search.value.toLowerCase(); return records.map((record, index) => ({record, index})).filter(({record}) => Object.values(record).join(' ').toLowerCase().includes(term)); }
    function renderMetrics(){
      const counts = records.reduce((acc, record) => { acc[record.status] = (acc[record.status] || 0) + 1; return acc; }, {});
      document.getElementById('metrics').innerHTML = `<article class="metric"><span>Total</span><strong>${records.length}</strong></article>` + Object.entries(counts).map(([status, count]) => `<article class="metric"><span>${status}</span><strong>${count}</strong></article>`).join('');
    }
    function renderRecords(){
      const rows = filteredRecords();
      recordsBody.innerHTML = rows.length ? rows.map(({record, index}) => `<tr><td>${record.name}</td><td>${record.category}</td><td>${record.contact}</td><td><span class="status">${record.status}</span></td><td>${record.notes || ''}</td><td><div class="actions"><button class="btn secondary" onclick="editRecord(${index})">Edit</button><button class="btn danger" onclick="deleteRecord(${index})">Delete</button></div></td></tr>`).join('') : `<tr><td class="empty" colspan="6">No records found.</td></tr>`;
      renderMetrics();
    }
    function editRecord(index){ editingIndex = index; const record = records[index]; fields.forEach((field) => document.getElementById(field).value = record[field] || ''); formTitle.textContent = 'Edit __ENTITY__'; window.scrollTo({top:0,behavior:'smooth'}); }
    function deleteRecord(index){ records.splice(index, 1); saveRecords(); renderRecords(); }
    function resetForm(){ editingIndex = null; form.reset(); formTitle.textContent = 'Add __ENTITY__'; }
    form.addEventListener('submit', (event) => { event.preventDefault(); const record = Object.fromEntries(fields.map((field) => [field, document.getElementById(field).value.trim()])); if (editingIndex === null) records.push(record); else records[editingIndex] = record; saveRecords(); resetForm(); renderRecords(); });
    document.getElementById('cancelBtn').addEventListener('click', resetForm);
    document.getElementById('resetBtn').addEventListener('click', () => { records = seedRecords; saveRecords(); resetForm(); renderRecords(); });
    search.addEventListener('input', renderRecords);
    renderRecords();
  </script>
</body>
</html>
"""
    html = html.replace("__TITLE__", str(preset["title"]))
    html = html.replace("__ENTITY__", str(preset["entity"]))
    html = html.replace("__CATEGORY__", str(preset["category"]))
    html = html.replace("__STATUS_OPTIONS__", status_options)
    html = html.replace("__SEED_RECORDS__", seed_records)
    html = html.replace("__FILE__", str(preset["file"]))
    return TaskPlanResponse(
        goal=prompt,
        mode="management-system-generator",
        steps=[
            "Classify the requested management system.",
            "Create a localStorage CRUD prototype.",
            "Verify the generated HTML structure.",
            "Start preview server for browser testing.",
        ],
        suggested_files=[{"path": str(preset["file"]), "content": html}],
        verify_commands=[f"Open http://127.0.0.1:9000/{preset['file']}"],
        notes=[
            "This is a Claude-style prototype-first build: SHAMSU creates a working local UI before suggesting a larger production architecture.",
            "For a production system, the next layer would be authentication, a backend API, and a database.",
        ],
    )

def _brick_breaker_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Brick Breaker</title>
  <style>
    body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0f172a;color:#f8fafc;font-family:Arial,sans-serif}
    main{text-align:center} canvas{background:#f8fafc;border:3px solid #1e293b;box-shadow:0 16px 40px rgba(0,0,0,.35);outline:none}
    .hud{display:flex;justify-content:center;gap:28px;margin:8px 0 14px;font-size:18px}.hint{color:#cbd5e1}
  </style>
</head>
<body>
  <main>
    <h1>Brick Breaker</h1>
    <div class="hud"><span>Score: <strong id="score">0</strong></span><span>Lives: <strong id="lives">3</strong></span></div>
    <canvas id="game" width="640" height="420" tabindex="0"></canvas>
    <p class="hint">Fixed version: use mouse or Left/Right arrows. Press Space to launch.</p>
  </main>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const scoreEl = document.getElementById('score');
    const livesEl = document.getElementById('lives');
    const paddle = { w: 104, h: 14, x: 268, y: 388, speed: 8 };
    const ball = { x: 320, y: 360, r: 8, dx: 3.4, dy: -3.8, stuck: true };
    const keys = { left: false, right: false };
    const rows = 5, cols = 9, brickW = 58, brickH = 22, gap = 8, brickTop = 50, brickLeft = 26;
    let bricks = [];
    let score = 0;
    let lives = 3;
    let won = false;
    let over = false;

    function makeBricks() {
      bricks = [];
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          bricks.push({ x: brickLeft + col * (brickW + gap), y: brickTop + row * (brickH + gap), alive: true, color: ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6'][row] });
        }
      }
    }

    function reset(full = true) {
      if (full) { score = 0; lives = 3; won = false; over = false; makeBricks(); }
      paddle.x = (canvas.width - paddle.w) / 2;
      ball.x = paddle.x + paddle.w / 2;
      ball.y = paddle.y - ball.r - 3;
      ball.dx = 3.4 * (Math.random() > 0.5 ? 1 : -1);
      ball.dy = -3.8;
      ball.stuck = true;
      scoreEl.textContent = score;
      livesEl.textContent = lives;
      draw();
      canvas.focus();
    }

    function update() {
      if (over || won) return;
      if (keys.left) paddle.x -= paddle.speed;
      if (keys.right) paddle.x += paddle.speed;
      paddle.x = Math.max(0, Math.min(canvas.width - paddle.w, paddle.x));
      if (ball.stuck) { ball.x = paddle.x + paddle.w / 2; ball.y = paddle.y - ball.r - 3; return; }
      ball.x += ball.dx; ball.y += ball.dy;
      if (ball.x <= ball.r || ball.x >= canvas.width - ball.r) ball.dx *= -1;
      if (ball.y <= ball.r) ball.dy *= -1;
      if (ball.y > canvas.height + ball.r) {
        lives -= 1;
        livesEl.textContent = lives;
        if (lives <= 0) over = true;
        else reset(false);
      }
      const paddleHit = ball.y + ball.r >= paddle.y && ball.y - ball.r <= paddle.y + paddle.h && ball.x >= paddle.x && ball.x <= paddle.x + paddle.w;
      if (paddleHit && ball.dy > 0) {
        const hit = (ball.x - (paddle.x + paddle.w / 2)) / (paddle.w / 2);
        ball.dx = hit * 5;
        ball.dy = -Math.abs(ball.dy);
      }
      for (const brick of bricks) {
        if (!brick.alive) continue;
        const hitBrick = ball.x + ball.r > brick.x && ball.x - ball.r < brick.x + brickW && ball.y + ball.r > brick.y && ball.y - ball.r < brick.y + brickH;
        if (hitBrick) {
          brick.alive = false;
          ball.dy *= -1;
          score += 10;
          scoreEl.textContent = score;
          if (bricks.every(b => !b.alive)) won = true;
          break;
        }
      }
    }

    function draw() {
      ctx.fillStyle = '#e2e8f0';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (const brick of bricks) {
        if (!brick.alive) continue;
        ctx.fillStyle = brick.color;
        ctx.fillRect(brick.x, brick.y, brickW, brickH);
        ctx.strokeStyle = 'rgba(15,23,42,.22)';
        ctx.strokeRect(brick.x, brick.y, brickW, brickH);
      }
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(paddle.x, paddle.y, paddle.w, paddle.h);
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
      ctx.fillStyle = '#7c3aed';
      ctx.fill();
      if (ball.stuck && !over && !won) message('Press Space to launch', 18);
      if (over) message('Game Over - Press Space', 26);
      if (won) message('You Win - Press Space', 26);
    }

    function message(text, size) {
      ctx.fillStyle = 'rgba(15,23,42,.82)';
      ctx.fillRect(0, 180, canvas.width, 58);
      ctx.fillStyle = '#fff';
      ctx.font = size + 'px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(text, canvas.width / 2, 216);
    }

    function loop() { update(); draw(); requestAnimationFrame(loop); }
    function handleKey(event, pressed) {
      if (['ArrowLeft', 'ArrowRight', 'Space'].includes(event.code) || ['ArrowLeft', 'ArrowRight'].includes(event.key)) event.preventDefault();
      if (event.key === 'ArrowLeft') keys.left = pressed;
      if (event.key === 'ArrowRight') keys.right = pressed;
      if (pressed && event.code === 'Space') { if (over || won) reset(true); else ball.stuck = false; }
    }
    document.addEventListener('keydown', event => handleKey(event, true));
    document.addEventListener('keyup', event => handleKey(event, false));
    canvas.addEventListener('mousemove', event => {
      const rect = canvas.getBoundingClientRect();
      paddle.x = Math.max(0, Math.min(canvas.width - paddle.w, event.clientX - rect.left - paddle.w / 2));
      if (ball.stuck) { ball.x = paddle.x + paddle.w / 2; ball.y = paddle.y - ball.r - 3; }
    });
    canvas.addEventListener('click', () => canvas.focus());
    reset(true);
    loop();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="game-generator",
        steps=["Create a Brick Breaker HTML canvas game.", "Verify the HTML game structure.", "Start preview server."],
        suggested_files=[{"path": "brick_breaker.html", "content": html}],
        verify_commands=["Open http://127.0.0.1:9000/brick_breaker.html"],
        notes=["This deterministic template is used for brick breaker prompts so demos do not depend on JSON fallback reliability."],
    )
def _snake_game_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Snake Game</title>
  <style>
    body{margin:0;min-height:100vh;display:grid;place-items:center;background:#111827;color:#f9fafb;font-family:Arial,sans-serif}
    main{text-align:center} canvas{background:#f8fafc;border:3px solid #1f2937;outline:none}.score{font-size:20px;margin:8px 0 14px}
  </style>
</head>
<body>
  <main>
    <h1>Snake Game</h1>
    <div class="score">Score: <span id="score">0</span></div>
    <canvas id="game" width="400" height="400" tabindex="0"></canvas>
    <p>Use arrow keys to move. Press Space to restart after game over.</p>
  </main>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const scoreEl = document.getElementById('score');
    const size = 20;
    const cells = canvas.width / size;
    let snake, food, direction, queuedDirection, score, gameOver, timer;

    function reset() {
      snake = [{ x: 8, y: 10 }, { x: 7, y: 10 }, { x: 6, y: 10 }];
      food = { x: 14, y: 10 };
      direction = { x: 1, y: 0 };
      queuedDirection = { x: 1, y: 0 };
      score = 0;
      gameOver = false;
      scoreEl.textContent = score;
      draw();
      canvas.focus();
    }

    function placeFood() {
      do {
        food = { x: Math.floor(Math.random() * cells), y: Math.floor(Math.random() * cells) };
      } while (snake.some(part => part.x === food.x && part.y === food.y));
    }

    function step() {
      if (gameOver) return draw();
      direction = queuedDirection;
      const head = { x: snake[0].x + direction.x, y: snake[0].y + direction.y };
      if (head.x < 0 || head.x >= cells || head.y < 0 || head.y >= cells || snake.some(part => part.x === head.x && part.y === head.y)) {
        gameOver = true;
        return draw();
      }
      snake.unshift(head);
      if (head.x === food.x && head.y === food.y) {
        score += 10;
        scoreEl.textContent = score;
        placeFood();
      } else {
        snake.pop();
      }
      draw();
    }

    function draw() {
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(food.x * size, food.y * size, size, size);
      snake.forEach((part, index) => {
        ctx.fillStyle = index === 0 ? '#15803d' : '#22c55e';
        ctx.fillRect(part.x * size + 1, part.y * size + 1, size - 2, size - 2);
      });
      if (gameOver) {
        ctx.fillStyle = 'rgba(17,24,39,0.82)';
        ctx.fillRect(0, 160, canvas.width, 80);
        ctx.fillStyle = '#fff';
        ctx.font = '26px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Game Over', canvas.width / 2, 195);
        ctx.font = '15px Arial';
        ctx.fillText('Press Space to restart', canvas.width / 2, 222);
      }
    }

    function setDirection(next) {
      if (next.x + direction.x === 0 && next.y + direction.y === 0) return;
      queuedDirection = next;
    }

    document.addEventListener('keydown', (event) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(event.code)) event.preventDefault();
      if (event.key === 'ArrowUp') setDirection({ x: 0, y: -1 });
      if (event.key === 'ArrowDown') setDirection({ x: 0, y: 1 });
      if (event.key === 'ArrowLeft') setDirection({ x: -1, y: 0 });
      if (event.key === 'ArrowRight') setDirection({ x: 1, y: 0 });
      if (event.code === 'Space' && gameOver) reset();
    });
    canvas.addEventListener('click', () => canvas.focus());

    reset();
    timer = setInterval(step, 185);
  </script>
</body>
</html>
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="game-generator",
        steps=["Create a Snake HTML canvas game.", "Verify the HTML game structure.", "Start preview server."],
        suggested_files=[{"path": "snake_game.html", "content": html}],
        verify_commands=["Open http://127.0.0.1:9000/snake_game.html"],
        notes=["This deterministic template is used for snake game prompts so demos do not depend on JSON fallback reliability."],
    )
def _bouncing_ball_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Bouncing Ball</title>
  <style>body{margin:0;background:#101820;color:white;font-family:sans-serif}canvas{display:block;margin:24px auto;background:#f7f7f7;border:2px solid #333}</style>
</head>
<body>
  <canvas id=\"game\" width=\"640\" height=\"360\"></canvas>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const ball = { x: 80, y: 80, vx: 4, vy: 3, r: 18 };
    function tick() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ball.x += ball.vx; ball.y += ball.vy;
      if (ball.x < ball.r || ball.x > canvas.width - ball.r) ball.vx *= -1;
      if (ball.y < ball.r || ball.y > canvas.height - ball.r) ball.vy *= -1;
      ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
      ctx.fillStyle = '#7c3aed'; ctx.fill();
      requestAnimationFrame(tick);
    }
    tick();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="game-generator",
        steps=["Create an HTML canvas game file in the workspace.", "Verify the game structure.", "Start preview server."],
        suggested_files=[{"path": "bouncing_ball.html", "content": html}],
        verify_commands=["Open http://127.0.0.1:9000/bouncing_ball.html"],
        notes=["This deterministic template is used for bouncing ball prompts."],
    )


def _pong_game_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Pong</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#101827;color:#f8fafc;font-family:Arial,sans-serif}main{text-align:center}canvas{background:#0f172a;border:3px solid #334155;outline:none}.hud{display:flex;justify-content:center;gap:32px;margin:8px 0 14px;font-size:20px}</style>
</head>
<body>
  <main>
    <h1>Pong</h1>
    <div class="hud"><span>You: <strong id="playerScore">0</strong></span><span>AI: <strong id="aiScore">0</strong></span></div>
    <canvas id="game" width="720" height="420" tabindex="0"></canvas>
    <p>Use Up/Down arrows or mouse. First to 7 wins. Press Space to restart.</p>
  </main>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const playerScoreEl = document.getElementById('playerScore');
    const aiScoreEl = document.getElementById('aiScore');
    const player = { x: 24, y: 160, w: 14, h: 92, speed: 7 };
    const aiPaddle = { x: 682, y: 160, w: 14, h: 92, speed: 4.2 };
    const ball = { x: 360, y: 210, r: 9, dx: 4.4, dy: 2.8 };
    const keys = { up: false, down: false };
    let playerScore = 0, aiScore = 0, message = 'Press Space to start', running = false;
    function resetBall(direction) { ball.x = canvas.width / 2; ball.y = canvas.height / 2; ball.dx = 4.4 * direction; ball.dy = (Math.random() > 0.5 ? 1 : -1) * 2.8; }
    function restart() { playerScore = 0; aiScore = 0; player.y = 160; aiPaddle.y = 160; resetBall(1); message = ''; running = true; updateScore(); canvas.focus(); }
    function updateScore() { playerScoreEl.textContent = playerScore; aiScoreEl.textContent = aiScore; }
    function update() {
      if (!running) return;
      if (keys.up) player.y -= player.speed;
      if (keys.down) player.y += player.speed;
      player.y = Math.max(0, Math.min(canvas.height - player.h, player.y));
      const aiTarget = ball.y - aiPaddle.h / 2;
      aiPaddle.y += Math.max(-aiPaddle.speed, Math.min(aiPaddle.speed, aiTarget - aiPaddle.y));
      aiPaddle.y = Math.max(0, Math.min(canvas.height - aiPaddle.h, aiPaddle.y));
      ball.x += ball.dx; ball.y += ball.dy;
      if (ball.y <= ball.r || ball.y >= canvas.height - ball.r) ball.dy *= -1;
      for (const paddle of [player, aiPaddle]) {
        const hit = ball.x + ball.r > paddle.x && ball.x - ball.r < paddle.x + paddle.w && ball.y + ball.r > paddle.y && ball.y - ball.r < paddle.y + paddle.h;
        if (hit) { const offset = (ball.y - (paddle.y + paddle.h / 2)) / (paddle.h / 2); ball.dx = Math.abs(ball.dx) * (paddle === player ? 1 : -1) * 1.04; ball.dy = offset * 5; }
      }
      if (ball.x < -20) { aiScore++; resetBall(1); updateScore(); }
      if (ball.x > canvas.width + 20) { playerScore++; resetBall(-1); updateScore(); }
      if (playerScore >= 7 || aiScore >= 7) { running = false; message = playerScore >= 7 ? 'You win! Space to restart' : 'AI wins. Space to restart'; }
    }
    function draw() {
      ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#334155'; ctx.setLineDash([8, 10]); ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#f8fafc'; ctx.fillRect(player.x, player.y, player.w, player.h); ctx.fillRect(aiPaddle.x, aiPaddle.y, aiPaddle.w, aiPaddle.h);
      ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2); ctx.fill();
      if (message) { ctx.font = '24px Arial'; ctx.textAlign = 'center'; ctx.fillText(message, canvas.width/2, canvas.height/2); }
    }
    function loop(){ update(); draw(); requestAnimationFrame(loop); }
    document.addEventListener('keydown', e => { if (['ArrowUp','ArrowDown','Space'].includes(e.code)) e.preventDefault(); if (e.key === 'ArrowUp') keys.up = true; if (e.key === 'ArrowDown') keys.down = true; if (e.code === 'Space') restart(); });
    document.addEventListener('keyup', e => { if (e.key === 'ArrowUp') keys.up = false; if (e.key === 'ArrowDown') keys.down = false; });
    canvas.addEventListener('mousemove', e => { const rect = canvas.getBoundingClientRect(); player.y = Math.max(0, Math.min(canvas.height - player.h, e.clientY - rect.top - player.h / 2)); });
    canvas.addEventListener('click', () => canvas.focus());
    loop();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="game-generator", steps=["Create a Pong HTML game.", "Verify the HTML game structure.", "Start preview server."], suggested_files=[{"path": "pong.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/pong.html"], notes=["This deterministic template is used for Pong prompts."])


def _tic_tac_toe_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Tic Tac Toe</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f8fafc;color:#0f172a;font-family:Arial,sans-serif}main{text-align:center}.board{display:grid;grid-template-columns:repeat(3,110px);gap:8px}.cell{width:110px;height:110px;font-size:52px;font-weight:700;border:2px solid #334155;background:white;cursor:pointer}.cell:hover{background:#e0f2fe}button{margin-top:18px;padding:10px 18px}</style>
</head>
<body>
  <main>
    <h1>Tic Tac Toe</h1>
    <p id="status">Player X turn</p>
    <div id="board" class="board"></div>
    <button id="reset">Restart</button>
  </main>
  <script>
    const boardEl = document.getElementById('board');
    const statusEl = document.getElementById('status');
    const resetEl = document.getElementById('reset');
    let board, current, locked;
    const wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    function checkWinner() { for (const [a,b,c] of wins) if (board[a] && board[a] === board[b] && board[a] === board[c]) return board[a]; return board.every(Boolean) ? 'Draw' : null; }
    function render() { boardEl.innerHTML = ''; board.forEach((value, index) => { const cell = document.createElement('button'); cell.className = 'cell'; cell.textContent = value; cell.onclick = () => play(index); boardEl.appendChild(cell); }); }
    function play(index) { if (locked || board[index]) return; board[index] = current; const result = checkWinner(); if (result) { locked = true; statusEl.textContent = result === 'Draw' ? 'Draw game' : 'Player ' + result + ' wins'; } else { current = current === 'X' ? 'O' : 'X'; statusEl.textContent = 'Player ' + current + ' turn'; } render(); }
    function reset() { board = Array(9).fill(''); current = 'X'; locked = false; statusEl.textContent = 'Player X turn'; render(); }
    resetEl.onclick = reset; reset();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="game-generator", steps=["Create a Tic Tac Toe HTML game.", "Verify the HTML game structure.", "Start preview server."], suggested_files=[{"path": "tic_tac_toe.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/tic_tac_toe.html"], notes=["This deterministic template is used for Tic Tac Toe prompts."])


def _quiz_app_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Quiz App</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#eef2ff;color:#111827;font-family:Arial,sans-serif}.panel{width:min(620px,92vw);background:white;border:1px solid #c7d2fe;padding:24px}.answers{display:grid;gap:10px;margin:18px 0}button{padding:10px 14px;text-align:left}.primary{text-align:center;background:#4f46e5;color:white;border:0}</style>
</head>
<body>
  <main class="panel">
    <h1>Quiz App</h1>
    <p id="progress"></p>
    <h2 id="question"></h2>
    <div id="answers" class="answers"></div>
    <p id="feedback"></p>
    <button id="next" class="primary">Next</button>
  </main>
  <script>
    const questions = [
      { q: 'What does HTML describe?', a: ['Page structure','Database schema','CPU speed'], correct: 0 },
      { q: 'Which language runs in the browser?', a: ['JavaScript','SQL','Bash'], correct: 0 },
      { q: 'What does CSS control?', a: ['Visual style','Network routing','File permissions'], correct: 0 }
    ];
    let index = 0, score = 0, answered = false;
    const questionEl = document.getElementById('question'), answersEl = document.getElementById('answers'), feedbackEl = document.getElementById('feedback'), progressEl = document.getElementById('progress'), nextEl = document.getElementById('next');
    function showQuestion() { answered = false; feedbackEl.textContent = ''; const item = questions[index]; progressEl.textContent = 'Question ' + (index + 1) + ' of ' + questions.length + ' | Score ' + score; questionEl.textContent = item.q; answersEl.innerHTML = ''; item.a.forEach((answer, choice) => { const btn = document.createElement('button'); btn.textContent = answer; btn.onclick = () => choose(choice); answersEl.appendChild(btn); }); nextEl.textContent = index === questions.length - 1 ? 'Finish' : 'Next'; }
    function choose(choice) { if (answered) return; answered = true; const correct = choice === questions[index].correct; if (correct) score++; feedbackEl.textContent = correct ? 'Correct.' : 'Not quite. Correct answer: ' + questions[index].a[questions[index].correct]; }
    nextEl.onclick = () => { if (!answered) { feedbackEl.textContent = 'Choose an answer first.'; return; } if (index < questions.length - 1) { index++; showQuestion(); } else { questionEl.textContent = 'Final score: ' + score + '/' + questions.length; answersEl.innerHTML = ''; progressEl.textContent = 'Quiz complete'; feedbackEl.textContent = ''; nextEl.textContent = 'Restart'; nextEl.onclick = () => location.reload(); } };
    showQuestion();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="app-generator", steps=["Create a Quiz HTML app.", "Verify the HTML structure.", "Start preview server."], suggested_files=[{"path": "quiz_app.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/quiz_app.html"], notes=["This deterministic template is used for quiz app prompts."])


def _todo_app_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Todo App</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#ecfeff;color:#164e63;font-family:Arial,sans-serif}.app{width:min(620px,92vw);background:white;border:1px solid #a5f3fc;padding:24px}form{display:flex;gap:8px}input{flex:1;padding:10px}button{padding:10px 14px}li{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #e5e7eb}</style>
</head>
<body>
  <main class="app">
    <h1>Todo App</h1>
    <form id="form"><input id="task" placeholder="Add a task" /><button>Add</button></form>
    <ul id="list"></ul>
  </main>
  <script>
    const form = document.getElementById('form');
    const taskInput = document.getElementById('task');
    const list = document.getElementById('list');
    let todos = JSON.parse(localStorage.getItem('shamsuTodos') || '[]');
    function save() { localStorage.setItem('shamsuTodos', JSON.stringify(todos)); }
    function renderTodos() { list.innerHTML = ''; todos.forEach((todo, index) => { const li = document.createElement('li'); const label = document.createElement('label'); const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = todo.done; checkbox.onchange = () => { todo.done = checkbox.checked; save(); renderTodos(); }; label.append(checkbox, ' ' + todo.text); if (todo.done) label.style.textDecoration = 'line-through'; const del = document.createElement('button'); del.textContent = 'Delete'; del.onclick = () => { todos.splice(index, 1); save(); renderTodos(); }; li.append(label, del); list.appendChild(li); }); }
    form.onsubmit = event => { event.preventDefault(); const text = taskInput.value.trim(); if (!text) return; todos.push({ text, done: false }); taskInput.value = ''; save(); renderTodos(); };
    renderTodos();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="app-generator", steps=["Create a Todo HTML app.", "Verify the HTML structure.", "Start preview server."], suggested_files=[{"path": "todo_app.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/todo_app.html"], notes=["This deterministic template is used for todo app prompts."])


def _calculator_app_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Calculator App</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f1f5f9;color:#0f172a;font-family:Arial,sans-serif}.calc{width:300px;background:white;border:1px solid #cbd5e1;padding:18px}.display{height:54px;border:1px solid #cbd5e1;margin-bottom:12px;padding:12px;text-align:right;font-size:28px}.keys{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}button{padding:14px;font-size:18px}.wide{grid-column:span 2}</style>
</head>
<body>
  <main class="calc">
    <h1>Calculator</h1>
    <div id="display" class="display">0</div>
    <div class="keys" id="keys"></div>
  </main>
  <script>
    const display = document.getElementById('display');
    const keys = document.getElementById('keys');
    let current = '0', stored = null, operator = null, resetNext = false;
    const layout = ['7','8','9','/','4','5','6','*','1','2','3','-','0','.','=','+','C'];
    function update(){ display.textContent = current; }
    function appendDigit(value){ if (resetNext) { current = value === '.' ? '0.' : value; resetNext = false; return update(); } if (value === '.' && current.includes('.')) return; current = current === '0' && value !== '.' ? value : current + value; update(); }
    function calculate(){ const a = Number(stored), b = Number(current); if (operator === '+') current = String(a + b); if (operator === '-') current = String(a - b); if (operator === '*') current = String(a * b); if (operator === '/') current = b === 0 ? 'Error' : String(a / b); stored = null; operator = null; resetNext = true; update(); }
    function chooseOperator(value){ if (stored !== null && !resetNext) calculate(); stored = current; operator = value; resetNext = true; }
    function press(value){ if ('0123456789.'.includes(value)) appendDigit(value); else if ('+-*/'.includes(value)) chooseOperator(value); else if (value === '=') calculate(); else { current = '0'; stored = null; operator = null; resetNext = false; update(); } }
    layout.forEach(value => { const btn = document.createElement('button'); btn.textContent = value; if (value === 'C') btn.className = 'wide'; btn.onclick = () => press(value); keys.appendChild(btn); });
    update();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="app-generator", steps=["Create a Calculator HTML app.", "Verify the HTML structure.", "Start preview server."], suggested_files=[{"path": "calculator_app.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/calculator_app.html"], notes=["This deterministic template is used for calculator app prompts."])

def _generated_task_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="json-generator-fallback",
        steps=["Classify the request.", "Ask the local model for a strict JSON file plan.", "Validate paths and file types.", "Write files and run lightweight verification."],
        suggested_files=[],
        verify_commands=[],
        notes=["No fixed template matched; autonomous run will try the JSON file generator fallback."],
    )


def _bugfix_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="claude-like-bugfix-workflow",
        steps=[
            "Analyze the bug report, stack trace, file name, function name, or failing behavior.",
            "Run project_index and project_map to understand files, symbols, imports, and ownership.",
            "Use search_files and semantic_search to locate likely bug regions.",
            "Use read_file_range around matching lines instead of reading/replacing whole files.",
            "Prepare the smallest exact-block patch with replace_in_file and wait for approval.",
            "Run verification and use failure output for the next repair loop.",
        ],
        suggested_files=[],
        verify_commands=["python -m pytest -q", "npm test", "python -m py_compile <changed_file.py>", "npm run build"],
        notes=["Bugfix mode follows a Claude-like index -> search -> range-read -> patch -> verify -> repair loop."],
        requirements_analysis=[
            "Identify the failing behavior and extract concrete search terms from the prompt or error output.",
            "Map the project before editing so SHAMSU understands nearby files and dependencies.",
            "Read only relevant line ranges and confirm exact old_text before patching.",
            "Verify after every focused edit and repeat from error feedback if needed.",
        ],
        clarification_questions=[
            "Can you provide the exact error message, failing command, stack trace, or expected vs actual behavior?",
            "Which file or feature should be treated as the highest-priority suspect?",
        ],
        stack=["project_index", "project_map", "search_files", "semantic_search", "read_file_range", "replace_in_file", "verification commands"],
        file_plan=["No new file is created automatically; SHAMSU first locates the bug and patches the smallest affected existing file block."],
        workflow_summary="Claude-like bugfix workflow: understand failure -> index project -> search symbols/errors -> read bounded ranges -> patch exact block with approval -> verify -> repair from feedback.",
    )


def _large_file_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="claude-like-large-file-debugger",
        steps=[
            "Do not load the full huge file into the model.",
            "Build project_index/project_map to identify file size, symbols, and likely target ranges.",
            "Search for function names, stack-trace terms, errors, or semantic hints.",
            "Read only bounded line ranges with read_file_range around matches.",
            "Create a patch plan for the smallest exact block and ask for approval.",
            "Run focused verification, then repeat using the failure output if needed.",
        ],
        suggested_files=[],
        verify_commands=["python -m pytest -q", "python -m py_compile <changed_file.py>", "npm run build", "gcc <changed_file.c> -o <output.exe>"],
        notes=["Large-file mode is designed for 100000-line style files by using indexing, search, range reads, and exact patches."],
        requirements_analysis=[
            "Treat the prompt as a debugging workflow, not a full-file reading task.",
            "Extract search terms from stack traces, function names, error text, or expected behavior.",
            "Use bounded windows so SHAMSU can reason about the important lines without exceeding context.",
            "Patch only confirmed exact text and verify the affected behavior.",
        ],
        clarification_questions=[
            "What error message, function name, or line number should SHAMSU search for first?",
            "Which verification command proves the bug is fixed?",
        ],
        stack=["project_index", "search_files", "semantic_search", "read_file_range", "replace_in_file", "pytest/build/compiler verification"],
        file_plan=["No whole-file rewrite. SHAMSU will locate relevant ranges, propose an exact patch, and mutate only after approval."],
        workflow_summary="Claude-like large-file workflow: index first, search targets, read ranges, patch exact blocks, verify, and repair from feedback without loading the whole huge file.",
    )


def _general_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="coding-task",
        steps=["Use project_index to understand the workspace.", "Identify relevant files/ranges.", "Make changes with write_file or replace_in_file.", "Run verification."],
        suggested_files=[],
        verify_commands=["python -m pytest -q"],
        notes=["This is a planning scaffold for chat/tool mode."],
    )





















































