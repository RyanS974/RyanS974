<!-- git tutorial, git.md -->
# Git Tutorial

This is a basic tutorial of git.

# Table of Contents

- [Overview of git](#overview-of-git)
- [git Basics]()
- [Working with Branches]()
- [Collaborating with git]()
- [Undoing Changes]()
- [git Workflows]()
- [git Commands]()
- [git Best Practices]()
- [How to Install git]()
- [GitHub]()
- [Web Sites of Interest]()

# Overview of Git

Git is a distributed version control system created by Linus Torvalds in 2005. It allows developers to track changes in source code during software development and supports non-linear development through its branching system.

Key concepts:
- Distributed system: Every developer has a full copy of the repository
- Snapshots: Git stores data as snapshots of the project over time
- Local operations: Most operations can be done offline
- Data integrity: All content is checksummed before storage

# Git Basics

Core concepts every Git user should understand:

1. **Repository (Repo)**: A directory where Git tracks changes to files
2. **Working Directory**: Where you modify files
3. **Staging Area**: Where you prepare changes for committing
4. **Commit**: A snapshot of your changes at a point in time

Basic workflow:
1. Modify files in working directory
2. Stage changes using `git add`
3. Commit changes using `git commit`
4. Push changes to remote using `git push`

# Working with Branches

Branches allow parallel development paths within a repository.

Common branch operations:
- Create branch: `git branch <branch-name>`
- Switch branch: `git checkout <branch-name>`
- Create and switch: `git checkout -b <branch-name>`
- List branches: `git branch`
- Delete branch: `git branch -d <branch-name>`
- Merge branches: `git merge <branch-name>`

# Collaborating with Git

Collaboration features:

1. **Remote Repositories**:
   - Clone: `git clone <url>`
   - Add remote: `git remote add <name> <url>`
   - Fetch changes: `git fetch`
   - Pull changes: `git pull`
   - Push changes: `git push`

2. **Pull Requests**:
   - Fork repositories
   - Create feature branches
   - Submit changes for review
   - Discuss and iterate
   - Merge approved changes

# Undoing Changes

Methods to undo changes:

1. **Unstaged Changes**:
   - Discard changes: `git checkout -- <file>`
   - Restore working directory: `git restore <file>`

2. **Staged Changes**:
   - Unstage changes: `git reset HEAD <file>`
   - Reset to commit: `git reset --hard <commit>`

3. **Commits**:
   - Amend last commit: `git commit --amend`
   - Revert commit: `git revert <commit>`
   - Reset to previous state: `git reset --hard HEAD~1`

# Git Workflows

Common Git workflows:

1. **Feature Branch Workflow**:
   - Create feature branch
   - Develop feature
   - Submit pull request
   - Review and merge

2. **Gitflow**:
   - Main branches: master, develop
   - Supporting branches: feature, release, hotfix
   - Structured release process

3. **Trunk-Based Development**:
   - Short-lived feature branches
   - Frequent integration to main branch
   - Emphasis on continuous integration


## Table of Commands

| Command | Description | Arguments |
|---------|-------------|-----------|
| `git init` | Initialize new repository | `[directory]` |
| `git clone` | Clone repository | `<repository> [directory]` |
| `git add` | Stage changes | `<pathspec>` or `.` for all |
| `git commit` | Record changes | `[-m <message>] [-a]` |
| `git push` | Upload changes | `[<remote>] [<branch>]` |
| `git pull` | Download and merge changes | `[<remote>] [<branch>]` |
| `git fetch` | Download changes without merging | `[<remote>] [<branch>]` |
| `git branch` | List, create, or delete branches | `[-d] [-D] [-r] [-a] [<branch-name>]` |
| `git checkout` | Switch branches or restore files | `[-b] <branch-name/file>` |
| `git switch` | Switch branches (new alternative to checkout) | `[-c] <branch-name>` |
| `git merge` | Combine branches | `<branch-name> [-no-ff]` |
| `git rebase` | Reapply commits on top of another base | `<branch-name> [-i]` |
| `git status` | Show working tree status | `[-s] [-b]` |
| `git log` | Show commit history | `[--oneline] [--graph] [--decorate]` |
| `git diff` | Show changes between commits | `[<commit>] [<file>]` |
| `git stash` | Stash changes | `[push] [pop] [apply] [list] [drop]` |
| `git reset` | Reset current HEAD | `[--soft|--mixed|--hard] [<commit>]` |
| `git revert` | Create new commit that undoes changes | `<commit>` |
| `git remote` | Manage remote repositories | `[-v] [add] [remove] [rename]` |
| `git fetch` | Download objects and refs | `[<remote>] [<branch>]` |
| `git tag` | Create, list, delete, or verify tags | `[-a] [-d] [-l] [<tagname>]` |
| `git cherry-pick` | Apply changes from existing commits | `<commit>...` |
| `git bisect` | Binary search for bugs | `start/bad/good/reset` |
| `git blame` | Show last modification to each line | `<file>` |
| `git grep` | Search working directory | `[-n] [-l] [-i] <pattern>` |
| `git clean` | Remove untracked files | `[-n] [-f] [-d] [-x]` |
| `git reflog` | Manage reflog information | `[show] [expire] [delete]` |
| `git rm` | Remove files from working tree | `[-f] [-r] [--cached] <file>` |
| `git mv` | Move or rename files | `<source> <destination>` |
| `git show` | Show various types of objects | `[<object>]` |
| `git config` | Get and set repository or global options | `[--global] [--system] <key> <value>` |
| `git restore` | Restore working tree files | `[--source=<tree>] [--staged] <pathspec>` |
| `git worktree` | Manage multiple working trees | `add/list/remove <path>` |
| `git submodule` | Initialize, update, or inspect submodules | `[init] [update] [status]` |
| `git archive` | Create archive of files | `--format=<fmt> --output=<file>` |
| `git describe` | Give object human readable name | `[--tags] [<commit-ish>]` |
| `git gc` | Cleanup unnecessary files | `[--aggressive] [--prune]` |
| `git fsck` | Verify database integrity | `[--full] [--strict]` |
| `git count-objects` | Count unpacked objects | `[-v] [-H]` |
| `git merge-base` | Find best common ancestor | `<commit1> <commit2>` |
| `git rev-parse` | Pick out and massage parameters | `[--short] [<rev>]` |
| `git shortlog` | Summarize git log output | `[-n] [-s] [-e]` |
| `git verify-commit` | Check GPG signature of commits | `<commit>...` |
| `git verify-tag` | Check GPG signature of tags | `<tag>...` |
| `git whatchanged` | Show logs with difference | `[<options>] [<since>..<until>]` |
| `git help` | Display help information | `[command] [--web]` |


Common Option Flags:
- `-f`: Force operation
- `-v`: Verbose output
- `-q`: Quiet operation
- `-n`: Dry run
- `-p`: Show patch
- `-b`: Show branch information
- `-d`: Delete
- `-m`: Message
- `-a`: All
- `-r`: Remote
- `--global`: Apply to all repositories
- `--cached`: Work with staged changes
- `--hard`: Reset working directory
- `--soft`: Keep working directory


# Git Best Practices

1. **Commit Practices**:
   - Write clear commit messages
   - Make atomic commits
   - Commit early and often
   - Use conventional commit formats

2. **Branch Management**:
   - Keep master/main branch stable
   - Delete merged branches
   - Use descriptive branch names
   - Regular rebasing to avoid conflicts

3. **Collaboration**:
   - Pull before pushing
   - Review code before merging
   - Use meaningful commit messages
   - Keep repos clean and organized

# How to Install Git

**Windows**:
1. Download Git from https://git-scm.com/download/win
2. Run installer with default options
3. Verify installation: `git --version`

**macOS**:
1. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Git: `brew install git`
3. Verify installation: `git --version`

**Linux (Ubuntu/Debian)**:
1. Update package list: `sudo apt update`
2. Install Git: `sudo apt install git`
3. Verify installation: `git --version`

# GitHub

GitHub is a web-based hosting service for Git repositories that adds collaboration features:

1. **Key Features**:
   - Repository hosting
   - Issue tracking
   - Pull requests
   - Project management
   - Actions (CI/CD)
   - Pages (static hosting)

2. **Getting Started**:
   - Create account
   - Set up SSH keys
   - Create/fork repositories
   - Clone repositories
   - Push changes

# Web Sites of Interest

- [Official Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git Branching Tutorial](https://learngitbranching.js.org/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials)
