"""Console output for `python -m en_gu registry`."""

from en_gu import registry


def show_registry(cfg, rollback=False):
    if rollback:
        version = registry.rollback(cfg)
        print(f"Rolled back: version {version.version} is the champion again")
    versions = registry.list_versions(cfg)
    if not versions:
        print(f"No versions of {registry.model_name(cfg)} yet.")
        return
    aliases = {}
    for alias in (registry.CHAMPION, registry.PREVIOUS, registry.CANDIDATE):
        found = registry.get_by_alias(cfg, alias)
        if found is not None:
            aliases.setdefault(str(found.version), []).append(alias)
    print(f"{registry.model_name(cfg)}")
    for v in versions:
        scores = registry.scores_of(v)
        shown = scores.get("test") or scores.get("reference") or {}
        chrf = f"chrF++ {shown['chrf']:.2f}" if "chrf" in shown else "no scores"
        gate = v.tags.get("gate")
        print(
            f"  v{v.version:<3} {chrf:<16} isolated={v.tags.get('eval_isolated', '?'):<6}"
            f"{('gate=' + gate) if gate else '':<13}{','.join(aliases.get(str(v.version), []))}"
        )
