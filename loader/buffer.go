package loader

import (
	"io/fs"
	"path/filepath"
)

func buffer(p string) []string {
	var err error

	var buf []string

	err = filepath.Walk(p, func(pat string, inf fs.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if pat == p {
			return nil
		}

		if !inf.IsDir() {
			return nil
		}

		{
			buf = append(buf, inf.Name())
		}

		if inf.IsDir() {
			return filepath.SkipDir
		}

		return nil
	})
	if err != nil {
		panic(err)
	}

	return buf
}