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
    f = context[buf]["ens"][subset].copy()
    l = f.pop(0)

    m = xgb.DMatrix(f, l)

    for buc in BUCKET:
      p.append(context[buf]["mod"][buc].predict(m, iteration_range=(0, context[buf]["mod"][buc].best_iteration + 1)))

  y_true = []

  for y in l:
    if y == 4:
      y_true.append(1)
    elif y < 4:
      y_true.append(2)
    elif y > 4:
      y_true.append(0)

  return xgb.DMatrix(pd.DataFrame(p).transpose(), pd.DataFrame(y_true))

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

def model_params(num_class=2):
  return {
    "booster": "gbtree",
    "grow_policy": "lossguide",
    "learning_rate": 0.02,
    "max_depth": 200,
    "objective": "multi:softmax",
    "num_class": num_class,
    "eval_metric": ["merror", "mlogloss"],
  }

################################################################################

def train_model(params, tra_mat, val_mat, evl_res=None):
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
  )

################################################################################

context = {}

################################################################################

context = fill_ens(context)
context = fill_mod(context)

################################################################################

labels = [-1, 0, +1]

################################################################################

tra_mat = build_ensemble_matrix(context, "tra")
tes_mat = build_ensemble_matrix(context, "tes")
val_mat = build_ensemble_matrix(context, "val")

################################################################################

ensemble = train_model(model_params(num_class=len(labels)), tra_mat, val_mat)

################################################################################

pre_mat = ensemble.predict(tes_mat, iteration_range=(0, ensemble.best_iteration + 1))

################################################################################

y_true = tes_mat.get_label()
y_pred = pre_mat

################################################################################

log_err = skl.metrics.mean_squared_log_error(y_true, y_pred)
print("log_err:", log_err)

################################################################################

pre_sco = skl.metrics.precision_score(y_true, y_pred, average="weighted")
print("pre_sco:", pre_sco)

################################################################################

ensemble.save_model("{{ .Pat }}" + "/ensemble.ubj")

################################################################################

pathlib.Path("{{ .Pat }}" + "/res/").mkdir(exist_ok=True)
with open("{{ .Pat }}" + "/res/res.json", 'a') as the_file:
    the_file.write(json.dumps({"log_err": log_err.astype(float), "pre_sco": pre_sco.astype(float)}) + '\n')
`
