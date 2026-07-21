# Git Flow Guidelines for AI Agent

You are an AI coding assistant that strictly follows the Git Flow branching model.  
Adhere to these rules for all version control operations unless explicitly instructed otherwise by the user.

---

## 1. Branch Architecture

### Permanent Branches
- **`main`** – Production-ready code. Only merged into via `release` or `hotfix` branches. Always deployable.
- **`develop`** – Integration branch for completed features. Latest development state.

### Temporary Branches
- **`feature/*`** – New features or enhancements.
- **`release/*`** – Preparation for a new production release.
- **`hotfix/*`** – Critical fixes for production issues.
- **`bugfix/*`** – Non-critical bug fixes (optional, can be treated as features).
- **`chore/*`** – Maintenance tasks (dependency updates, config changes).

---

## 2. Branch Naming Conventions

Use **kebab-case**, lowercase alphanumeric and hyphens only. Include a ticket reference if available.

```
feature/<short-description>          # feature/add-user-auth
feature/JIRA-123-add-user-auth       # with ticket
bugfix/<description>                 # bugfix/fix-login-timeout
release/<semver>                     # release/1.2.0
hotfix/<semver>                      # hotfix/1.1.1
chore/<description>                  # chore/update-dependencies
```

- `<short-description>` must be 2–5 words, no verbs as first word if redundant (e.g., avoid `feature/add-login-feature` → use `feature/login` or `feature/add-login`).
- Do not use prefixes like `feature/feature-`, no double nesting.

---

## 3. Workflow – Step by Step

### 3.1 Starting a Feature
1. Always branch off from `develop`.
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/<name>
   ```
2. Commit frequently with [Conventional Commits](#4-commit-message-format).
3. Keep the branch focused on a single concern. If scope creeps, discuss splitting.

### 3.2 Keeping a Feature Up-to-Date
- Rebase onto `develop` regularly to avoid massive merge conflicts:
  ```bash
  git fetch origin develop
  git rebase origin/develop
  ```
- Resolve conflicts carefully, keeping the feature’s intent intact.
- Never rebase a branch that has been pushed and is used by others unless agreed. In that case, prefer merging `develop` into the feature.

### 3.3 Finishing a Feature
1. Ensure all tests pass and the branch is rebased onto the latest `develop`.
2. Push the branch and create a Pull Request (PR) targeting `develop`.
3. After approval, **squash-merge** (or merge with a merge commit, per team policy).  
   Prefer squash-merge to keep a linear history on `develop`.
4. Delete the remote feature branch after merge.

### 3.4 Creating a Release
1. When `develop` reflects the desired state for a new release, create a release branch:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/<version>
   ```
   `<version>` must follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH` (e.g., `1.2.0`).
2. On this branch, only bump version numbers, update changelogs, fix minor release-blocking bugs. No new features.
3. Once ready, merge into `main` **and** back into `develop`:
   ```bash
   git checkout main
   git merge --no-ff release/<version>
   git tag -a v<version> -m "Release <version>"
   git push origin main --tags

   git checkout develop
   git merge --no-ff release/<version>
   git push origin develop
   ```
   The `--no-ff` flag ensures a merge commit is always created, preserving the release as a historical checkpoint.
4. Delete the release branch.

### 3.5 Hotfixes
1. Branch off from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/<version>
   ```
   Hotfix version is the next patch number: e.g., if latest tag is `v1.2.0`, hotfix is `v1.2.1`.
2. Fix the issue and commit. Bump the patch version.
3. Merge into both `main` and `develop` (same `--no-ff` process as release):
    - Merge to `main`, tag with `v<version>`, push tags.
    - Merge to `develop` (resolve any conflicts, preferring hotfix changes).
4. Delete the hotfix branch.

---

## 4. Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) strictly.

```
<type>(<optional scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Allowed types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

- **feat** – A new feature (triggers a MINOR version bump).
- **fix** – A bug fix (triggers a PATCH version bump).
- **BREAKING CHANGE** – Mention in footer, or use `!` after type/scope: `feat!:` (triggers MAJOR bump).
- **Short summary** ≤ 72 characters, imperative mood (“add” not “added”).
- Reference tickets in footer: `Closes #123` or `Refs JIRA-456`.

**Examples:**
```
feat(auth): add magic link login

Implement passwordless email authentication.
Closes #42
```
```
fix(api): handle null response from payment gateway

The gateway sometimes returns a null body, causing a crash.
Now defaults to empty object.
```
```
chore(deps): bump requests from 2.28.0 to 2.31.0
```

---

## 5. Merging & Pull Request Rules

- **Always create a PR** for integrating feature/hotfix/release branches. Never push directly to `main` or `develop`.
- **PR title must match the conventional commit format** (it will become the merge commit message if squashing).
- **Require at least one approval** (unless hotfix and emergency protocol is in place).
- The AI agent may propose a PR description summarizing changes, but final merging requires human approval.
- Merge strategy:
    - Feature → `develop`: **squash and merge** (clean linear history).
    - Release → `main` / `develop`: **merge commit** (`--no-ff`).
    - Hotfix → `main` / `develop`: **merge commit** (`--no-ff`).

---

## 6. Tagging and Versioning

- Tags are only created on the `main` branch, for releases and hotfixes.
- Tag format: `v<MAJOR>.<MINOR>.<PATCH>`, e.g., `v2.1.0`.
- Annotated tags preferred: `git tag -a v1.0.0 -m "Release v1.0.0: initial production release"`.
- Never move or delete a tag after pushing, unless an emergency and communicated team-wide.

---

## 7. Handling Conflicts

- When rebasing or merging, the AI agent must:
    - Clearly present the conflicting sections to the user.
    - Explain the context (which branch’s change is which).
    - Propose a resolution, but **never auto-resolve unless explicitly instructed**.
- After resolution, ensure the project still builds / passes checks.

---

## 8. AI Agent-Specific Rules

- **Do not force-push** to shared branches (`main`, `develop`, `release/*`, `hotfix/*`). Only force-push to your own feature branch if rebasing and you have confirmed it’s safe.
- Before starting any git operation, check the current branch and status (`git status`). If the working directory is dirty, ask the user to commit or stash changes.
- When asked to “deploy” or “release”, confirm the target environment and the version before creating the release branch.
- Maintain a `.gitignore` and never commit build artifacts, `.env` files, or IDE-specific folders unless explicitly configured.
- If a user requests an action that deviates from these guidelines (e.g., direct commit to `main`), the agent must warn about the violation and suggest the correct flow.
- Provide clear, step-by-step commands when guiding a user through the flow.

---

## 9. Quick Reference: Commands for Agent

```bash
# Start feature
git checkout develop && git pull origin develop
git checkout -b feature/<name>

# Finish feature (after rebase and PR merge)
git branch -d feature/<name>   # local cleanup

# Start release
git checkout develop && git pull origin develop
git checkout -b release/1.2.0
# ... bump version, update changelog, final tests
git checkout main && git merge --no-ff release/1.2.0
git tag -a v1.2.0 -m "Release 1.2.0"
git push origin main --tags
git checkout develop && git merge --no-ff release/1.2.0
git push origin develop
git branch -d release/1.2.0

# Start hotfix
git checkout main && git pull origin main
git checkout -b hotfix/1.2.1
# ... fix and bump version
git checkout main && git merge --no-ff hotfix/1.2.1
git tag -a v1.2.1 -m "Hotfix 1.2.1"
git push origin main --tags
git checkout develop && git merge --no-ff hotfix/1.2.1
git push origin develop
git branch -d hotfix/1.2.1
```
