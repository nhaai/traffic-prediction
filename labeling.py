import json

with open("camera_config.json", "r") as f:
    CAM_CFG = json.load(f)

def label_by_density(cd: float, rules: dict):
    congested_rule = rules.get("congested") or {}
    moderate_rule = rules.get("moderate") or {}
    free_rule = rules.get("free") or {}

    cd_min_cong = congested_rule.get("crowd_density_min")
    cd_min_mod = moderate_rule.get("crowd_density_min")
    cd_max_mod = moderate_rule.get("crowd_density_max")
    cd_max_free = free_rule.get("crowd_density_max")

    if cd_min_cong is not None and cd >= cd_min_cong:
        return "congested"

    if cd_max_free is not None and cd <= cd_max_free:
        return "free_flow"

    if cd_min_mod is not None and cd_max_mod is not None:
        if cd_min_mod <= cd <= cd_max_mod:
            return "moderate"

    return None

def rule_match(feats: dict, rule_dict: dict) -> bool:
    """
    Generic matching:
      <name>_min -> feats[name] >= value
      <name>_max -> feats[name] <= value
      <name>     -> feats[name] >= value
    """
    for key, value in rule_dict.items():
        if key.endswith("_min"):
            field = key[:-4]  # remove "_min"
            if feats.get(field, 0) < value:
                return False
        elif key.endswith("_max"):
            field = key[:-4]  # remove "_max"
            if feats.get(field, 0) > value:
                return False
        else:
            # bare key: treat as minimum threshold
            field = key
            if feats.get(field, 0) < value:
                return False

    return True

def auto_label(feats: dict, cam_id: str) -> str:
    """
    Priority order:
      1. If camera has rule: check rule with crowd_density included
      2. If rule does NOT match: fallback to crowd_density threshold
      3. If crowd_density missing: fallback to classical rules
    """
    if cam_id not in CAM_CFG:
        raise KeyError(f"Camera '{cam_id}' not found")
    rules = CAM_CFG[cam_id].get("rules", {})

    congested_rule = rules.get("congested")
    free_rule = rules.get("free")
    moderate_rule = rules.get("moderate")

    cd_raw = feats.get("crowd_density")
    cd = float(cd_raw) if cd_raw is not None else -1.0
    has_cd = cd_raw is not None

    # crowd density rule
    if has_cd and congested_rule:
        cd_min = congested_rule.get("crowd_density_min")
        if cd_min is not None and cd >= cd_min:
            return "congested"

    if has_cd and free_rule:
        cd_max = free_rule.get("crowd_density_max")
        if cd_max is not None and cd <= cd_max:
            return "free_flow"

    if has_cd and moderate_rule:
        cd_min = moderate_rule.get("crowd_density_min")
        cd_max = moderate_rule.get("crowd_density_max")
        if (cd_min is None or cd >= cd_min) and (cd_max is None or cd <= cd_max):
            return "moderate"

    # feature-based rule
    if congested_rule and rule_match(feats, congested_rule):
        return "congested"
    if free_rule and rule_match(feats, free_rule):
        return "free_flow"
    if moderate_rule and rule_match(feats, moderate_rule):
        return "moderate"

    # fallback: no density or no rule match
    if has_cd:
        if cd < 6:
            return "free_flow"
        elif cd < 14:
            return "moderate"
        else:
            return "congested"

    return "moderate"
