package ensemble

const deftem = `
import json
import pathlib

import numpy as np
import pandas as pd
import sklearn as skl
import xgboost as xgb

################################################################################

BUFFER = [
{{- range $b := .Buf }}
    "{{ $b }}",
{{- end }}
]

BUFFER.sort()

################################################################################

BUCKET = [
{{- range $b := .Buc }}
    "{{ $b }}",
{{- end }}
]

################################################################################

def build_ensemble_matrix(context, subset):
  l = {}
  p = []

  for buf in BUFFER:
    f = context[buf]["ens"][subset].copy().astype('float')
    l = f.pop(0)

    m = xgb.DMatrix(f, l)

    for buc in BUCKET:
      pre = context[buf]["mod"][buc].predict(m, iteration_range=(0, context[buf]["mod"][buc].best_iteration + 1))
      p.append(normalize(pre))

  y_true = []

  for y in l:
    if y == 5:
      y_true.append(0.5)
    elif y < 5:
      y_true.append(1.0)
    elif y > 5:
      y_true.append(0.0)

  return xgb.DMatrix(pd.DataFrame(p).transpose(), pd.DataFrame(y_true))

################################################################################

def ensemble_params():
  return {
    "base_score": 0.50,
    "booster": "gbtree",
    "gamma": 10.00,
    "grow_policy": "lossguide",
    "learning_rate": 0.02,
    "max_depth": 20,
    "objective": "reg:logistic",
    "eval_metric": ["error", "logloss"],
  }

################################################################################

def fill_ens(context):
  for buf in BUFFER:
    context[buf] = {
        "ens": {
            "tra": pd.read_csv("{{ .Pat }}" + "/" + buf + "/ens/tra.csv", header=None),
            "tes": pd.read_csv("{{ .Pat }}" + "/" + buf + "/ens/tes.csv", header=None),
            "val": pd.read_csv("{{ .Pat }}" + "/" + buf + "/ens/val.csv", header=None),
        }
    }

  return context

################################################################################

def fill_mod(context):
  for buf in BUFFER:
    context[buf]["mod"] = {}

    for buc in BUCKET:
      context[buf]["mod"][buc] = load_model("{{ .Pat }}" + "/" + buf + "/mod/" + buc + ".ubj")

  return context

################################################################################

def load_model(p):
  m = xgb.Booster()

  m.load_model(p)

  return m

################################################################################

def normalize(l):
  l = np.where(l > 0.85, 1, l)
  l = np.where(((l >= 0.15) & (l <= 0.85)), 0.5, l)
  l = np.where(l < 0.15, 0, l)

  return l

################################################################################

def train_model(params, tra_mat, val_mat, evl_res=None, xgb_mod=None):
  return xgb.train(
    params,
    tra_mat,
    num_boost_round=5000,
    callbacks=[
        xgb.callback.EarlyStopping(rounds=25),
    ],
    evals=[(tra_mat, 'tra_mat'), (val_mat, 'val_mat')],
    evals_result=evl_res,
    verbose_eval=100,
    xgb_model=xgb_mod,
  )

################################################################################

context = {}

################################################################################

context = fill_ens(context)
context = fill_mod(context)

################################################################################

tra_mat = build_ensemble_matrix(context, "tra")
tes_mat = build_ensemble_matrix(context, "tes")
val_mat = build_ensemble_matrix(context, "val")

################################################################################

ensemble = train_model(
  ensemble_params(),
  tra_mat,
  val_mat,
{{- if .Upd }}
  xgb_mod="{{ .Pat }}" + "/ensemble.ubj",
{{- end }}
)

################################################################################

pre_mat = ensemble.predict(tes_mat, iteration_range=(0, ensemble.best_iteration + 1))

################################################################################

y_true = tes_mat.get_label()
y_pred = normalize(pre_mat)

################################################################################

log_err = skl.metrics.mean_squared_log_error(y_true, y_pred)
print("log_err:", log_err)

################################################################################

ensemble.save_model("{{ .Pat }}" + "/ensemble.ubj")

################################################################################

pathlib.Path("{{ .Pat }}" + "/res/").mkdir(exist_ok=True)
with open("{{ .Pat }}" + "/res/res.json", 'w') as the_file:
    the_file.write(json.dumps({"log_err": log_err.astype(float)}) + '\n')
`
