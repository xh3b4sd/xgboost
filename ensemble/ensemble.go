package ensemble

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

type Ensemble struct {
	// Buc is the required bucket list.
	Buc []string
	// Buf is the required list of buffer hashes for training this ensemble.
	Buf []string
	Cmd *exec.Cmd
	Deb bool
	Fil *os.File
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
}

func (e *Ensemble) Execute() ([]byte, error) {
	{
		e.configs()
	}

	var buf bytes.Buffer
	{
		t, err := template.New("ensemble").Parse(e.Tem)
		if err != nil {
			return nil, tracer.Mask(err)
		}

		err = t.Execute(&buf, e.mapping())
		if err != nil {
			return nil, tracer.Mask(err)
		}
	}

	return buf.Bytes(), nil
}

func (e *Ensemble) Train() error {
	var err error

	{
		e.configs()
		e.cleanup()
	}

	var byt []byte
	{
		byt, err = e.Execute()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		e.Fil, err = os.CreateTemp("", "xgboost-ensemble-template-*")
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		_, err := e.Fil.Write(byt)
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		err := e.Fil.Close()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		err = ioutil.WriteFile(e.temfilp(), e.temfilb(), 0664)
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		e.Cmd = exec.Command("python3", e.Fil.Name())
	}

	if e.Deb {
		e.Cmd.Stdout = os.Stdout
		e.Cmd.Stderr = os.Stderr
	}

	{
		err := e.Cmd.Start()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		err := e.Cmd.Wait()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		e.cleanup()
	}

	return nil
}

func (e *Ensemble) cleanup() {
	if exists(e.temfilp()) {
		if exists(e.temfilc()) {
			err := os.Remove(e.temfilc())
			if err != nil {
				panic(err)
			}
		}

		err := os.Remove(e.temfilp())
		if err != nil {
			panic(err)
		}
	}
}

func (e *Ensemble) configs() {
	if len(e.Buc) == 0 {
		panic("Ensemble.Buc must not be empty")
	}

	if len(e.Buf) == 0 {
		panic("Ensemble.Buf must not be empty")
	}

	if e.Pat == "" {
		panic("Ensemble.Pat must not be empty")
	}

	if e.Tem == "" {
		e.Tem = deftem
	}
}

func (e *Ensemble) mapping() map[string]interface{} {
	return map[string]interface{}{
		"Buc": e.Buc,
		"Buf": e.Buf,
		"Pat": strings.TrimSuffix(e.Pat, "/"),
	}
}

func (e *Ensemble) temfilb() []byte {
	return []byte(e.Fil.Name())
}

func (e *Ensemble) temfilc() string {
	byt, err := ioutil.ReadFile(e.temfilp())
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(string(byt))
}

func (e *Ensemble) temfilp() string {
	return filepath.Join(e.Pat, "ensemble.pat")
}
