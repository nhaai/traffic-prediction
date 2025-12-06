import json

with open("camera_config.json", "r") as f:
    CAM_CFG = json.load(f)

def rule_match(feats: dict, rule_dict: dict) -> bool:
    """
    Convention for keys in rule_dict:
      "<name>_min"         -> feats["<name>"] >= value
      "<name>_max"         -> feats["<name>"] <= value
      "<name>"             -> feats["<name>"] >= value  (generic threshold)
    Example (from JSON):
      total_min            -> feats["total"]
      density_mid_min      -> feats["density_mid"]
      bbox_ratio_min       -> feats["bbox_ratio"]
      top_min              -> feats["top"]
      motor_min            -> feats["motor"]
      bottom_motor_min     -> feats["bottom_motor"]
      mid_car_min          -> feats["mid_car"]
      cluster_density_min  -> feats["cluster_density"]
      bottom_min           -> feats["bottom"]
      top_max              -> feats["top"]
      total_max            -> feats["total"]
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
    if cam_id not in CAM_CFG:
        raise KeyError(f"Camera '{cam_id}' not found in camera_config.json")
    rules = CAM_CFG[cam_id].get("rules", {})

    congested_rule = rules.get("congested")
    moderate_rule = rules.get("moderate")
    free_rule = rules.get("free")

    # Check congested
    if congested_rule and rule_match(feats, congested_rule):
        return "congested"

    # Check free-flow (usually defined by max_* constraints)
    if free_rule and rule_match(feats, free_rule):
        return "free_flow"

    # Check moderate
    if moderate_rule and rule_match(feats, moderate_rule):
        return "moderate"

    # Fallback: treat as moderate if no rule matches
    return "moderate"
