/**
 * Pre-baked sample transactions extracted from the held-out test set.
 * Each one demonstrates a different model behaviour for the demo UI.
 *
 * Source: src/extract_samples_one_off.py (see Phase 4.2 in the project's git log).
 */
import type { Transaction } from "./types";

export interface TransactionSample {
  id: "genuine" | "fraud" | "borderline";
  label: string;
  description: string;
  actualClass: 0 | 1;
  expectedFraudProbability: number;
  transaction: Transaction;
}

export const SAMPLES: TransactionSample[] = [
  {
    id: "genuine",
    label: "Genuine sample",
    description:
      "Clear-cut legitimate purchase. The model scores it near zero — confident genuine.",
    actualClass: 0,
    expectedFraudProbability: 0.000001,
    transaction: {
      Time: 160760, Amount: 23,
      V1: -0.674466, V2: 1.408105, V3: -1.110622, V4: -1.328366, V5: 1.388996,
      V6: -1.308439, V7: 1.885879, V8: -0.614233, V9: 0.311652, V10: 0.650757,
      V11: -0.857785, V12: -0.229961, V13: -0.199817, V14: 0.266371, V15: -0.046544,
      V16: -0.741398, V17: -0.605617, V18: -0.392568, V19: -0.162648, V20: 0.394322,
      V21: 0.080084, V22: 0.810034, V23: -0.224327, V24: 0.707899, V25: -0.135837,
      V26: 0.045102, V27: 0.533837, V28: 0.291319,
    },
  },
  {
    id: "fraud",
    label: "Fraud sample",
    description:
      "Card-testing fraud: a $0.01 probe purchase with extreme V-values. The model catches it with 99.98% confidence.",
    actualClass: 1,
    expectedFraudProbability: 0.999798,
    transaction: {
      Time: 57007, Amount: 0.01,
      V1: -1.271244, V2: 2.462675, V3: -2.851395, V4: 2.32448, V5: -1.372245,
      V6: -0.948196, V7: -3.065234, V8: 1.166927, V9: -2.268771, V10: -4.881143,
      V11: 2.255147, V12: -4.686387, V13: 0.652375, V14: -6.174288, V15: 0.59438,
      V16: -4.849692, V17: -6.536521, V18: -3.119094, V19: 1.715494, V20: 0.560478,
      V21: 0.652941, V22: 0.081931, V23: -0.221348, V24: -0.523582, V25: 0.224228,
      V26: 0.756335, V27: 0.6328, V28: 0.250187,
    },
  },
  {
    id: "borderline",
    label: "Borderline sample",
    description:
      "A LEGITIMATE $89.99 purchase that the model scores ~0.098 — above our chosen 0.035 threshold. This is exactly the kind of false positive threshold tuning trades for higher recall.",
    actualClass: 0,
    expectedFraudProbability: 0.098418,
    transaction: {
      Time: 28818, Amount: 89.99,
      V1: -2.181281, V2: 3.422398, V3: -6.628457, V4: 4.096195, V5: -2.558534,
      V6: -2.902141, V7: -4.627348, V8: 1.78316, V9: -2.689058, V10: -6.959551,
      V11: 4.146474, V12: -8.203205, V13: -1.415738, V14: -10.00665, V15: 1.065094,
      V16: -5.502404, V17: -8.902106, V18: -2.752053, V19: 1.024737, V20: 0.79216,
      V21: 0.890441, V22: -0.286284, V23: 0.049511, V24: -0.063265, V25: -0.158701,
      V26: -0.297131, V27: 1.34846, V28: 0.330177,
    },
  },
];
