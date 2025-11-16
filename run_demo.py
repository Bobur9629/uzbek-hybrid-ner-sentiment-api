# src/run_demo.py

from pprint import pprint

from .hybrid_model import HybridUzbekModel, HybridUzbekModelV01


def main():
    print("Hybrid Uzbek Model demo")
    print("Bo'sh satr kiritsang – dastur tugaydi.\n")

    model_rules = HybridUzbekModelV01()
    model_full = HybridUzbekModel()

    while True:
        text = input("Matn kiriting: ").strip()
        if not text:
            print("Chiqildi.")
            break

        print("\n--- Rules-only natija ---")
        res_rules = model_rules.analyze(text)
        pprint(res_rules)

        print("\n--- Full hybrid (ML-sentiment + rules-NER) natija ---")
        res_full = model_full.analyze(text)
        pprint(res_full)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
