package xgboost

type Fitter interface {
	Models()
	Single()
}

// Loader describes how boosters can be restored and used for predictions.
//
//     ldr := &loader.Loader{
//         Pat: "/Users/xh3b4sd/dat/",
//         Por: 8080,
//         Tem: "<template.py>",
//     }
//
type Loader interface {
	// Restore loads an ensemble of layered models originally saved in universal
	// binary JSON format via the "save_model" Python API of XGBoost. Suppose
	// having trained and saved 3 models in Python, namely a, b and c.
	//
	//     a.save_model("a.ubj") // buffer hash foo
	//     b.save_model("b.ubj") // buffer hash bar
	//     c.save_model("c.ubj") // buffer hash baz
	//
	// The 3 models saved above feed predictions into a combining ensemble e
	// which we have trained and saved in Python as well.
	//
	//     e.save_model("e.ubj")
	//
	// A booster instance implementing the Loader interface based on our
	// scenario described above can spawn a child process given the required
	// data path, server port and script template.
	//
	//     err := ldr.Restore()
	//
	// For more information on the IO Model of XGBoost see their official
	// documentation.
	//
	//     https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
	//
	Restore() error
	// Predict can be called to gather predictions from the underlying ensemble
	// once the ensemble instance got bootstrapped via Restore. As per our
	// example from above, the input data would be a mapping of inputs per
	// underlying model, where the map keys would resemble the buffer hashes of
	// the trained models.
	//
	//     inp := map[string][]float32{
	//         "foo": [ ... ], // features for a.ubj
	//         "bar": [ ... ], // features for b.ubj
	//         "baz": [ ... ], // features for c.ubj
	//     }
	//
	// For a multiclass ensemble the returned prediction should yield the
	// predicted class in numeric representation.
	Predict(map[string][]float32) (float32, error)
	// Sigkill shuts down the spawned child process. No predictions can be made
	// anymore after calling Sigkill.
	Sigkill() error
}
