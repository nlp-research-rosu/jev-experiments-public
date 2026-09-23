def test_factorial_views_change_only_the_declared_axes():
    from experiments.contrast_format_probe import probe_suite

    source = probe_suite(nested=True, heldout_wording=True)
    flat = probe_suite(nested=False, heldout_wording=True)
    wording = probe_suite(nested=True, heldout_wording=False)
    assert len(source["cases"]) == len(flat["cases"]) == len(wording["cases"]) == 16
    for a, b, c in zip(source["cases"], flat["cases"], wording["cases"], strict=True):
        assert a["expected"] == b["expected"] == c["expected"]
        assert a["request"]["questions"] == b["request"]["questions"]
        assert a["request"]["state"] == c["request"]["state"]
        nested = a["request"]["state"]
        structured = b["request"]["state"]
        assert nested["record"] == {k: v for k, v in structured.items() if k != "task"}
        assert all(str(v) in nested["request_brief"] for v in structured["task"].values())
        for qid in a["expected"]:
            assert a["request"]["questions"][qid]["criteria"] == c["request"]["questions"][qid]["criteria"]
