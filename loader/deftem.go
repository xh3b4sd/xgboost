package loader

const deftem = `
import json

import pandas as pd
import xgboost as xgb

from http.server import BaseHTTPRequestHandler, HTTPServer

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

def build_ensemble_matrix(context):
  p = []

  for buf in BUFFER:
    m = xgb.DMatrix(context[buf]["ens"])

    for buc in BUCKET:
      p.append(context[buf]["mod"][buc].predict(m, iteration_range=(0, context[buf]["mod"][buc].best_iteration + 1)))

  return xgb.DMatrix(pd.DataFrame(p).transpose())

################################################################################

def fill_ens(context, input):
  for buf in BUFFER:
    f = pd.DataFrame([input[buf]]).copy()
    f.pop(0)
    context[buf]["ens"] = f.astype('float')

  return context

################################################################################

def fill_mod(context):
  for buf in BUFFER:
    context[buf] = {
        "mod": {},
    }

    for buc in BUCKET:
      context[buf]["mod"][buc] = load_model("{{ .Pat }}" + "/" + buf + "/mod/" + buc + ".ubj")

  context["ens"] = load_model("{{ .Pat }}/ensemble.ubj")

  return context

################################################################################

def load_model(p):
  m = xgb.Booster()

  m.load_model(p)

  return m

################################################################################

context = fill_mod({})

################################################################################

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_response()
        self.wfile.write("OK\n".encode("utf-8"))

    def do_POST(self):
        con_len = int(self.headers.get('Content-Length'))
        req_bod = json.loads(self.rfile.read(con_len).decode('utf-8'))
        tes_mat = build_ensemble_matrix(fill_ens(context, req_bod))
        pre_mat = context["ens"].predict(tes_mat, iteration_range=(0, context["ens"].best_iteration + 1))

        self._set_response()
        self.wfile.write(f'{pre_mat[0]:.1f}'.encode())

    def log_message(self, format, *args):
        return

################################################################################

def run(server_class=HTTPServer, handler_class=S, addr="{{ .Add }}", port={{ .Por }}):
    httpd = server_class((addr, port), handler_class)
    print('Starting http server')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print('Stopping http server')

################################################################################

run()
`
