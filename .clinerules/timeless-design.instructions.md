# Timeless Design Instructions

## Overview

Design documents must be timeless - they capture current canonical understanding without time dependencies. This ensures documents remain relevant and maintainable as the project evolves.

## Core Principles

### 1. No Time-Bound Language

**❌ Avoid:**

- Dates in titles or content (e.g., "2025-01-05-updates")
- Version references (e.g., "v2.0 changes")
- Temporal qualifiers ("new", "updated", "corrected", "recent")
- Historical context ("previously", "before", "old implementation")

**✅ Use:**

- Present tense for current design
- Descriptive, timeless titles
- Canonical explanations without historical framing

### 2. No Update Mentality

**❌ Avoid:**

- "Key Updates" or "Changes" sections
- "Problem Solved" framing
- "Previous implementation had issues" language
- Changelog-style content in design docs

**✅ Use:**

- Current design presented as authoritative
- Natural document evolution through editing
- Focus on "what is" not "what was changed"

### 3. Timeless File Organization

**❌ Avoid:**

- Numbered files requiring renumbering (01-, 02-, etc.)
- Date-stamped filenames
- Versioned directories

**✅ Use:**

- Descriptive, semantic filenames
- Logical grouping by topic
- Files that don't need renaming when new ones are added

### 4. Canonical Content Structure

**❌ Avoid:**

- Time-specific sections
- "As of [date]" qualifiers
- References to external events or timelines

**✅ Use:**

- Current best practices
- Established patterns and conventions
- Forward-looking design decisions

## Examples

### ❌ Time-Bound (Avoid)

```markdown
## Key Updates (2025-01-05)

### New Inflation Algorithm

**Problem Solved**: Previous proximity pinning didn't work for real clothing.

**Updated Implementation:**

- Added center-outward inflation
- Fixed spring bone gravity physics
```

### ✅ Timeless (Preferred)

```markdown
## Garment Fitting Algorithm

**Center-Outward Inflation**: Clothing expands from body center outward for natural draping.

**Implementation:**

- Four-phase inflation process
- Semantic anchor points
- Physics-based spring forces
```

## File Naming Convention

### ❌ Time-Bound Names

- `01-system-architecture.md`
- `2025-01-05-pipeline-updates.md`
- `v2.0-api-changes.md`

### ✅ Timeless Names

- `system-architecture.md`
- `pipeline-stages.md`
- `api-design.md`

## Maintenance Guidelines

### When Updating Documents

1. **Edit in place** - Don't create "update" documents
2. **Remove obsolete content** - Don't keep historical versions
3. **Use descriptive commits** - Explain what changed, not when
4. **Preserve logical flow** - Maintain document coherence

### When Adding New Documents

1. **Choose semantic names** - Reflect document purpose
2. **Check for conflicts** - Ensure names don't collide
3. **Update references** - Fix links in other documents
4. **Maintain organization** - Keep related docs together

## Quality Checklist

- [ ] No dates in titles or content
- [ ] No "updated/corrected/new" qualifiers
- [ ] No historical "problem solved" framing
- [ ] Descriptive, stable filenames
- [ ] Current design presented as canonical
- [ ] No time-dependent references

## Rationale

Timeless design documents:

- **Remain relevant** indefinitely
- **Reduce maintenance overhead**
- **Focus on current understanding**
- **Enable natural evolution**
- **Avoid artificial versioning**

Design documents should capture enduring knowledge, not transient updates.
