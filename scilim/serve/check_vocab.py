"""Is JSD well-defined between two models? (check_vocab_match.py)

What must match is the id->token MAPPING, not how each tokenizer encodes a
sentence. Exit 0 = run JSD, 2 = do not.
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--max_mismatch_pct", type=float, default=5.0)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    s = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True)
    t = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
    print(f"student vocab {s.vocab_size} | teacher vocab {t.vocab_size}")
    if s.vocab_size != t.vocab_size:
        print("VERDICT: vocab sizes differ — do NOT run JSD with this pair.")
        sys.exit(2)
    sv, tv = s.get_vocab(), t.get_vocab()
    if sv != tv:
        inv_s, inv_t = {v: k for k, v in sv.items()}, {v: k for k, v in tv.items()}
        bad = [i for i in range(s.vocab_size) if inv_s.get(i) != inv_t.get(i)]
        pct = 100 * len(bad) / s.vocab_size
        print(f"{len(bad)} id slots ({pct:.2f}%) map to different tokens (masked from JSD)")
        for i in bad[:5]:
            print(f"   id {i}: student={inv_s.get(i)!r} teacher={inv_t.get(i)!r}")
        if pct > a.max_mismatch_pct:
            print("VERDICT: too much disagreement — do NOT run JSD.")
            sys.exit(2)
    probe = "The paper reports BLEU 31.2 on WMT14 without error bars or ablations."
    if s(probe)["input_ids"] != t(probe)["input_ids"]:
        print("note: encoding policy differs on the probe sentence (harmless; one tokenizer is used for both)")
    print("VERDICT: run the JSD stage.")
    sys.exit(0)


if __name__ == "__main__":
    main()
