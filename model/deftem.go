package model

const deftem = `
import json
import pathlib

import numpy as np
import pandas as pd
import sklearn as skl
import xgboost as xgb

################################################################################

BUFFER = "{{ .Buf }}"

################################################################################

context = {
{{- range $b := .Buc }}
    "{{ $b }}": {},
{{- end }}
}

################################################################################

def build_ensemble_matrix(context, path):
  c = pd.read_csv(path, header=None)

  f = c.copy().astype('float')
  l = f.pop(0)

  x = xgb.DMatrix(f, l)
  p = []

  for k, v in context.items():
    pre = v["mod"].predict(x, iteration_range=(0, v["mod"].best_iteration + 1))
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

def build_model_matrix(path):
  fea = []
  lab = []

  for p in path:
    c = pd.read_csv(p, header=None)

    f = c.copy().astype('float')
    l = f.pop(0)

    fea.append(f)
    lab.append(l)

  return xgb.DMatrix(pd.concat(fea, axis=0, ignore_index=True), pd.concat(lab, axis=0, ignore_index=True))

################################################################################

def create_models(context):
  for k, v in context.items():
    v["mod"].save_model("{{ .Pat }}" + "/" + BUFFER + "/mod/" + k + ".ubj")

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

def model_params():
  return {
    "base_score": 0.01,
    "booster": "gbtree",
    "gamma": 10.00,
    "grow_policy": "lossguide",
    "learning_rate": 0.02,
    "max_depth": 20,
    "objective": "reg:logistic",
    "eval_metric": ["error", "logloss"],
  }

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

for k, v in context.items():
  context[k]["tra_mat"] = build_model_matrix(["{{ .Pat }}" + "/" + BUFFER + "/csv/" + k + ".tra.csv"])
  context[k]["tes_mat"] = build_model_matrix(["{{ .Pat }}" + "/" + BUFFER + "/csv/" + k + ".tes.csv"])
  context[k]["val_mat"] = build_model_matrix(["{{ .Pat }}" + "/" + BUFFER + "/csv/" + k + ".val.csv"])

################################################################################

for k, v in context.items():
  print("train model " + k)
  context[k]["mod"] = train_model(
    model_params(),
    context[k]["tra_mat"],
    context[k]["val_mat"],
{{- if .Upd }}
    xgb_mod="{{ .Pat }}" + "/" + BUFFER + "/mod/" + k + ".ubj",
{{- end }}
  )

################################################################################

tra_mat = build_ensemble_matrix(context, "{{ .Pat }}" + "/" + BUFFER + "/ful/tra.csv")
tes_mat = build_ensemble_matrix(context, "{{ .Pat }}" + "/" + BUFFER + "/ful/tes.csv")
val_mat = build_ensemble_matrix(context, "{{ .Pat }}" + "/" + BUFFER + "/ful/val.csv")

################################################################################

print("train ensemble")
ensemble = train_model(ensemble_params(), tra_mat, val_mat)

################################################################################

pre_mat = ensemble.predict(tes_mat, iteration_range=(0, ensemble.best_iteration + 1))

################################################################################

y_true = tes_mat.get_label()
y_pred = normalize(pre_mat)

################################################################################

log_err = skl.metrics.mean_squared_log_error(y_true, y_pred)
print("log_err:", log_err)

################################################################################
{{ if .Upd }}
create_models(context)
{{- else }}
if log_err < {{ .Log }}:
  create_models(context)
{{- end }}

################################################################################

pathlib.Path("{{ .Pat }}" + "/" + BUFFER + "/res/").mkdir(exist_ok=True)
with open("{{ .Pat }}" + "/" + BUFFER + "/res/res.json", 'w') as the_file:
    the_file.write(json.dumps({"log_err": log_err.astype(float)}) + '\n')
`
