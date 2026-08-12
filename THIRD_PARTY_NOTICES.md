# Third-Party Notices

## Current status

**No third-party code has been copied or adapted into this repository.**

Phase 0 vendors nothing. The dependency list is empty apart from development tools, which are ordinary PyPI dependencies and are not vendored.

This file exists from the first commit so that the review process is in place before there is any pressure to skip it.

---

## Reuse policy

The project is Apache-2.0 and wants to reuse good code where that is legally and technically appropriate. Two rules govern how.

### Rule 1: reuse is evaluated at the point of need, not in advance

Do not survey other projects and pull pieces from them before the component that needs them exists. A repository assembled from six unrelated codebases inherits six sets of assumptions about storage, identifiers, and error handling, and those assumptions will not agree.

The order is: build the component's interface and a working naive implementation first, then look at prior art to replace or improve the internals.

Review schedule, by component:

| Component | Review when starting | Prior art to read |
| --- | --- | --- |
| Compaction | Phase 5 and Phase 8 | OpenClaw, Magic Compact |
| Provenance | Phase 12 | OpenChronicle |
| Retrieval and memory | Phase 10 and Phase 21 | Mem0, LycheeMemory, Memori |
| Snapshots and rollback | Phase 22 | Memoria |
| Paging and working set | Phase 20 | Pichay |
| Portable export | Phase 18 | PAM and other portable memory formats |
| Filesystem and Markdown state | Phase 12 onward | Acontext |

Reading a project for its ideas carries no license obligation and needs no entry in this file. Copying or adapting its code does.

### Rule 2: a public repository is not a license

Before any code is copied or adapted:

1. Read the repository's actual `LICENSE` file. A README that says "open source" is not evidence.
2. Read file-level copyright headers on the specific files involved. They can differ from the repository license.
3. Read any `NOTICE` file and honor it.
4. Check the transitive dependency licenses of anything the copied code pulls in.
5. Confirm compatibility with Apache-2.0.
6. Preserve required copyright and license notices verbatim.
7. Add an entry to this file before merging.
8. Record the upstream repository, the commit SHA, the exact files, and every modification made.
9. Copy nothing from a repository with unclear, missing, or conflicting licensing.
10. Prefer a small isolated adaptation over lifting a subsystem.

If a license is ambiguous, reimplement the idea independently. Ideas are not copyrightable; expression is.

Do not copy trademarks, logos, project names, or branding assets under any circumstance. Those are governed by trademark law, not by the source license, and a permissive code license grants no rights to them.

### License compatibility, for reference

| Upstream license | Compatible with Apache-2.0 project | Obligation if copied |
| --- | --- | --- |
| MIT | Yes | Preserve copyright line and license text |
| BSD-2 and BSD-3 | Yes | Preserve copyright line and license text |
| Apache-2.0 | Yes | Preserve license, `NOTICE` contents, and change notices |
| ISC | Yes | Preserve copyright line and license text |
| MPL-2.0 | File-level, with care | Modified MPL files stay MPL and must be disclosed |
| GPL and AGPL | No | Do not copy. Reimplement independently |
| Unlicensed or unclear | No | Do not copy |

---

## Entry template

Every future reuse gets a block in this shape. Do not abbreviate it.

```
## <Project name>

Repository: <url>
Commit:     <full SHA>
License:    <SPDX identifier>
Retrieved:  <YYYY-MM-DD>

Files here:
  src/open_context/<path>

Upstream files:
  <path in upstream repository>

What was taken:
  <adapted code | verbatim code | data or fixtures>

Modifications:
  <specific list, not "various changes">

Notices preserved at:
  <path to the retained copyright header or license copy>
```

Only code that was actually copied or adapted belongs here. Architectural inspiration does not, and listing it as reuse overstates the obligation and muddies the record.
