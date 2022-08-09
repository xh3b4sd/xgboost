package ensemble

import "os"

func exists(file string) bool {
	_, err := os.Stat(file)
	if os.IsNotExist(err) {
		return false
	} else if err != nil {
		panic(err)
	}

	return true
}
