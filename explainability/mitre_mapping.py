from typing import Dict, Any, List

def evaluate_mitre_context(features: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Evaluates network features using rule-based heuristics to suggest possible MITRE ATT&CK context.
    
    Args:
        features (Dict[str, float]): The raw network state features.
        
    Returns:
        List[Dict[str, Any]]: A list of possible MITRE contexts.
    """
    contexts = []
    
    # Extract common features (safe fallback if missing)
    syn_cnt = features.get("SYN Flag Cnt", 0.0)
    rst_cnt = features.get("RST Flag Cnt", 0.0)
    dst_port = features.get("Dst Port", 0.0)
    pkts_s = features.get("Flow Pkts/s", 0.0)
    fwd_len = features.get("TotLen Fwd Pkts", 0.0)
    bwd_len = features.get("TotLen Bwd Pkts", 0.0)
    flow_dur = features.get("Flow Duration", 1.0)
    
    # 1. Reconnaissance / Scanning (High SYN or high packet rate)
    if syn_cnt > 5 or pkts_s > 10000:
        contexts.append({
            "stage": "Reconnaissance",
            "confidence": "Medium",
            "evidence": f"High SYN Flag Count ({syn_cnt}) and/or Flow Pkts/s ({pkts_s:.2f})",
            "reason": "Rapid connection attempts or high SYN packets often indicate port scanning or reconnaissance."
        })
        
    # 2. Initial Access (Targeting vulnerable or common access ports)
    vulnerable_ports = [21, 22, 23, 3389, 445]
    if dst_port in vulnerable_ports:
        contexts.append({
            "stage": "Initial Access",
            "confidence": "Low",
            "evidence": f"Destination Port: {int(dst_port)}",
            "reason": "Traffic targeting common remote access or vulnerable service ports (e.g., SSH, RDP, SMB)."
        })
        
    # 3. Impact / DoS (Extremely high packet rate, short duration, high RST)
    if pkts_s > 50000 or rst_cnt > 10:
        contexts.append({
            "stage": "Impact (Denial of Service)",
            "confidence": "High",
            "evidence": f"Flow Pkts/s ({pkts_s:.2f}), RST Flag Count ({rst_cnt})",
            "reason": "Abnormally high packet rates or connection resets strongly correlate with resource exhaustion or flood attacks."
        })
        
    # 4. Exfiltration (Large asymmetric data transfer)
    # E.g., large outbound compared to inbound, or just huge transfer.
    # Assuming Fwd is typically client->server (outbound from internal)
    if fwd_len > 1_000_000 and fwd_len > 10 * bwd_len:
        contexts.append({
            "stage": "Exfiltration",
            "confidence": "Medium",
            "evidence": f"Large asymmetric forward payload (Fwd: {fwd_len} bytes, Bwd: {bwd_len} bytes)",
            "reason": "Unusual volume of outbound data relative to inbound responses suggests potential data staging or exfiltration."
        })

    # If no specific patterns match but we want a generic response
    if not contexts:
        contexts.append({
            "stage": "Unknown",
            "confidence": "None",
            "evidence": "No clear heuristic match",
            "reason": "The network flow does not exhibit explicit heuristic signatures for a specific MITRE ATT&CK stage."
        })
        
    # Always append the honest disclaimer
    for ctx in contexts:
        ctx["disclaimer"] = "Possible MITRE ATT&CK context based on flow heuristics. Flow-only data cannot definitively prove an attack technique."

    return contexts
