package loader

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/xh3b4sd/tracer"
)

type Loader struct {
	Add string
	// Buc is the required bucket list.
	Buc []string
	Cli *http.Client
	Cmd *exec.Cmd
	Deb bool
	Fil *os.File
	// Pat is the required data path containing all ensemble data.
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
	// Por is the required free port number used to run a simple HTTP server in
	// Python for serving predictions between processes.
	Por int
	// Tem is the required Python script template that is first being rendered
	// and persisted, and then executed in a child process.
	Tem string
	Url string
}

func (l *Loader) Execute() ([]byte, error) {
	{
		l.configs()
	}

	var buf bytes.Buffer
	{
		t, err := template.New("loader").Parse(l.Tem)
		if err != nil {
			return nil, tracer.Mask(err)
		}

		err = t.Execute(&buf, l.mapping())
		if err != nil {
			return nil, tracer.Mask(err)
		}
	}

	return buf.Bytes(), nil
}

func (l *Loader) Restore() error {
	var err error

	{
		l.configs()
		l.cleanup()
	}

	var byt []byte
	{
		byt, err = l.Execute()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		l.Fil, err = os.CreateTemp("", "xgboost-loader-template-*")
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		_, err := l.Fil.Write(byt)
		if err != nil {
			return tracer.Mask(err)
		}
	}

	{
		l.Cmd = exec.Command("python3", l.Fil.Name())
	}

	if l.Deb {
		l.Cmd.Stdout = os.Stdout
		l.Cmd.Stderr = os.Stderr
	}

	{
		err := l.Cmd.Start()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	go func() {
		err := l.Cmd.Wait()
		if err != nil {
			fmt.Printf("%s - %s\n", l.Fil.Name(), err.Error())
		}
	}()

	{
		err := l.Fil.Close()
		if err != nil {
			return tracer.Mask(err)
		}
	}

	for {
		if l.checker() {
			break
		}

		{
			time.After(3 * time.Second)
		}
	}

	{
		err = ioutil.WriteFile(l.pidfilp(), l.pidfilb(), 0664)
		if err != nil {
			return tracer.Mask(err)
		}
		err = ioutil.WriteFile(l.temfilp(), l.temfilb(), 0664)
		if err != nil {
			return tracer.Mask(err)
		}
	}

	return nil
}

func (l *Loader) Predict(inp map[string][]float32) (float32, error) {
	var err error

	var byt []byte
	{
		byt, err = json.Marshal(inp)
		if err != nil {
			return 0, tracer.Mask(err)
		}
	}

	var req *http.Request
	{
		req, err = http.NewRequest("POST", l.Url, bytes.NewBuffer(byt))
		if err != nil {
			return 0, tracer.Mask(err)
		}
	}

	{
		req.Header.Set("Content-Type", "application/json")
	}

	var res *http.Response
	{
		res, err = l.Cli.Do(req)
		if err != nil {
			return 0, tracer.Mask(err)
		}
		defer res.Body.Close()
	}

	var bod []byte
	{
		bod, err = ioutil.ReadAll(res.Body)
		if err != nil {
			return 0, tracer.Mask(err)
		}
	}

	var flo float64
	{
		flo, err = strconv.ParseFloat(string(bod), 32)
		if err != nil {
			return 0, tracer.Mask(err)
		}
	}

	return float32(flo), nil
}

func (l *Loader) Sigkill() error {
	var err error

	err = l.Cmd.Process.Kill()
	if err != nil {
		return tracer.Mask(err)
	}

	err = os.Remove(l.Fil.Name())
	if err != nil {
		return tracer.Mask(err)
	}

	err = os.Remove(l.pidfilp())
	if err != nil {
		return tracer.Mask(err)
	}

	err = os.Remove(l.temfilp())
	if err != nil {
		return tracer.Mask(err)
	}

	return nil
}

func (l *Loader) checker() bool {
	var err error

	var req *http.Request
	{
		req, err = http.NewRequest("GET", l.Url, nil)
		if err != nil {
			panic(err)
		}
	}

	var res *http.Response
	{
		res, err = l.Cli.Do(req)
		if err != nil {
			return false
		}
		defer res.Body.Close()
	}

	var bod []byte
	{
		bod, err = ioutil.ReadAll(res.Body)
		if err != nil {
			panic(err)
		}
	}

	return strings.TrimSpace(string(bod)) == "OK"
}

func (l *Loader) cleanup() {
	if exists(l.pidfilp()) {
		pro, err := os.FindProcess(l.pidfilc())
		if err != nil {
			panic(err)
		}

		err = pro.Kill()
		if IsProcessAlreadyFinished(err) {
			// fall through
		} else if err != nil {
			panic(err)
		}

		for {
			// In case a pidfile was found we deal with an orphaned process that
			// will never be a child process of the current process inspecting the
			// orphan. Process.Wait will therefore always return an error, which we
			// simply ignore. All we care about is the exit information of the
			// orphan.
			//
			//     wait: no child processes
			//
			sta, _ := pro.Wait()
			if sta == nil || sta.Exited() {
				break
			}

			{
				time.After(3 * time.Second)
			}
		}

		err = os.Remove(l.pidfilp())
		if err != nil {
			panic(err)
		}
	}

	if exists(l.temfilc()) {
		err := os.Remove(l.temfilc())
		if err != nil {
			panic(err)
		}
	}

	if exists(l.temfilp()) {
		err := os.Remove(l.temfilp())
		if err != nil {
			panic(err)
		}
	}
}

func (l *Loader) configs() {
	if l.Add == "" {
		l.Add = "localhost"
	}

	if len(l.Buc) == 0 {
		panic("Loader.Buc must not be empty")
	}

	if l.Cli == nil {
		l.Cli = &http.Client{}
	}

	if l.Pat == "" {
		panic("Loader.Pat must not be empty")
	}

	if l.Por == 0 {
		panic("Loader.Por must not be empty")
	}

	if l.Tem == "" {
		l.Tem = deftem
	}

	if l.Url == "" {
		l.Url = fmt.Sprintf("http://%s:%d", l.Add, l.Por)
	}
}

func (l *Loader) mapping() map[string]interface{} {
	return map[string]interface{}{
		"Add": l.Add,
		"Buc": l.Buc,
		"Buf": buffer(l.Pat),
		"Pat": strings.TrimSuffix(l.Pat, "/"),
		"Por": l.Por,
	}
}

func (l *Loader) pidfilb() []byte {
	return []byte(fmt.Sprintf("%d\n", l.Cmd.Process.Pid))
}

func (l *Loader) pidfilc() int {
	byt, err := ioutil.ReadFile(l.pidfilp())
	if err != nil {
		panic(err)
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(byt)))
	if err != nil {
		panic(err)
	}

	return pid
}

func (l *Loader) pidfilp() string {
	return filepath.Join(l.Pat, "loader.pid")
}

func (l *Loader) temfilb() []byte {
	return []byte(l.Fil.Name())
}

func (l *Loader) temfilc() string {
	byt, err := ioutil.ReadFile(l.temfilp())
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(string(byt))
}

func (l *Loader) temfilp() string {
	return filepath.Join(l.Pat, "loader.pat")
}
