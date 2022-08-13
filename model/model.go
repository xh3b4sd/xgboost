package model

import (
	"bytes"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/xh3b4sd/tracer"
)

type Model struct {
	// Buc is the required bucket list.
	Buc []string
	// Buf is the required buffer hash for training this model.
	Buf string
	Cmd *exec.Cmd
	Deb bool
	Fil *os.File
	// Log is the required maximum logarithmic error a trained model must not
	// exceed in order to be considered valid.
	Log float32
	// Pat is the required data path in which the data of the trained model will
	// be put in.
	//
	//     $ tree -L 1 /Users/xh3b4sd/dat/
	//     /Users/xh3b4sd/dat/
	//     ├── 01f5d6a195c0e829bdaee3ba3103159b
	//     ├── 2b26756ea93b780dcd4c0f2a2d97e21c
	//     ├── 6972b8a2c5236906a7e5de63a4ff9e16
	//     ├── 6aea549b64115f5a74207d68134ca69f
	//     ├── 6f8a6eab2a1b74cde85383adb8d99279
	//     ├── 7f1ef35480f2d65b88ae06412ef300b0
	//     ├── 99820be31a90480afdebf72ea186a33d
	//     ├── a31ab3bd91d4e13a7d12a76df1a9162c
	//     ├── ad74d526afa54106c98b820492c34fe2
	//     ├── d35366d6297711088fd486795fd7cc7a
	//     └── ensemble.ubj
	//
	Pat string
	// Tem is the required Python script template that is first being rendered
	// and persisted, and then executed in a child process.
	Tem string
	// Upd requires a model to exist in order for it to continue training on the
	// prepared data set.
	Upd bool
}

func (m *Model) Execute() ([]byte, error) {
	{
		m.configs()
	}

	var buf bytes.Buffer
	{
		t, err := template.New("model").Parse(m.Tem)
		if err != nil {
			return nil, tracer.Mask(err)
		}

		err = t.Execute(&buf, m.mapping())
		if err != nil {
			return nil, tracer.Mask(err)
		}
	}

	return buf.Bytes(), nil
}

func (m *Model) Train() error {
	var err error

	{
		m.configs()
		m.cleanup()
	}

	var byt []byte
	{
		byt, err = m.Execute()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		m.Fil, err = os.CreateTemp("", "xgboost-model-template-*")
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		_, err := m.Fil.Write(byt)
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		err := m.Fil.Close()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		err = ioutil.WriteFile(m.temfilp(), m.temfilb(), 0664)
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		m.Cmd = exec.Command("python3", m.Fil.Name())
	}

	if m.Deb {
		m.Cmd.Stdout = os.Stdout
		m.Cmd.Stderr = os.Stderr
	}

	{
		err := m.Cmd.Start()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		err := m.Cmd.Wait()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		m.cleanup()
	}

	return nil
}

func (m *Model) cleanup() {
	if exists(m.temfilp()) {
		if exists(m.temfilc()) {
			err := os.Remove(m.temfilc())
			if err != nil {
				panic(err)
			}
		}

		err := os.Remove(m.temfilp())
		if err != nil {
			panic(err)
		}
	}
}

func (m *Model) configs() {
	if len(m.Buc) == 0 {
		panic("Model.Buc must not be empty")
	}

	if m.Buf == "" {
		panic("Model.Buf must not be empty")
	}

	if m.Log == 0 {
		panic("Model.Log must not be empty")
	}

	if m.Pat == "" {
		panic("Model.Pat must not be empty")
	}

	if m.Tem == "" {
		m.Tem = deftem
	}
}

func (m *Model) mapping() map[string]interface{} {
	return map[string]interface{}{
		"Buc": m.Buc,
		"Buf": m.Buf,
		"Log": m.Log,
		"Pat": strings.TrimSuffix(m.Pat, "/"),
		"Upd": m.Upd,
	}
}

func (m *Model) temfilb() []byte {
	return []byte(m.Fil.Name())
}

func (m *Model) temfilc() string {
	byt, err := ioutil.ReadFile(m.temfilp())
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(string(byt))
}

func (m *Model) temfilp() string {
	return filepath.Join(m.Pat, m.Buf, "model.pat")
}
