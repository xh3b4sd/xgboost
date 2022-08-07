package loader

import (
	"errors"
	"os"
)

func IsProcessAlreadyFinished(err error) bool {
	return errors.Is(err, os.ErrProcessDone)
}
