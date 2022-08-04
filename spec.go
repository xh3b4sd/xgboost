package xgboost

// Ensemble describes how boosters can be restored and used for predictions.
// Creating a new ensemble might be as simple as shown below.
//
//     ens := &Booster{}
//
type Ensemble interface {
	// Restore loads an ensemble of layered models originally saved in universal
	// binary JSON format via the "save_model" Python API of XGBoost. Suppose
	// having trained and saved 3 models in Python, namely a, b and c.
	//
	//     a.save_model("a.ubj")
	//     b.save_model("b.ubj")
	//     c.save_model("c.ubj")
	//
	// The 3 models saved above feed predictions into a combining ensemble e
	// which we have trained and saved in Python as well.
	//
	//     e.save_model("e.ubj")
	//
	// A booster instance implementing the Ensemble interface based on our
	// scenario described above can spawn a child process given the file paths
	// to the respective model artefacts.
	//
	//     err := ens.Restore("e.ubj", "a.ubj", "b.ubj", "c.ubj")
	//
	// For more information on the IO Model of XGBoost see their official
	// documentation.
	//
	//     https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
	//
	Restore(string, ...string) error
	// Predict can be called to gather predictions from the underlying ensemble
	// once the ensemble instance got bootstrapped via Restore. As per our
	// example from above, the input data would be lists of inputs per
	// underlying model, ordered in training order.
	//
	//     inp := [][]float32{
	//         { ... }, // input for a.ubj
	//         { ... }, // input for b.ubj
	//         { ... }, // input for c.ubj
	//     }
	//
	// For a multiclass ensemble the returned prediction should yield the
	// predicted class in numeric representation.
	Predict([][]float32) (float32, error)
	// Sigkill shuts down the spawned child process. No predictions can be made
	// anymore after calling Sigkill.
	Sigkill() error
}
