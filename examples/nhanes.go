package main

// Some examples of fitting GLM's to NHANES data.
//
// To prepare, download the demographics (DEMO_G.XPT) and blood
// pressure (BPX_G.XPT) data from here:
//
// https://wwwn.cdc.gov/Nchs/Nhanes/Search/DataPage.aspx?Component=Examination&CycleBeginYear=2011
//
// I don't know of a Go reader for SAS XPT files.  This script is set
// up to use a merged dataset in csv format.  This can be accomplised
// using the following Python script:
//
// # Python script below, requires Pandas
// import pandas as pd
//
// fn1 = "DEMO_G.XPT"
// fn2 = "BPX_G.XPT"
//
// ds1 = pd.read_sas(fn1)
// ds2 = pd.read_sas(fn2)
//
// ds = pd.merge(ds1, ds2, left_on="SEQN", right_on="SEQN")
//
// ds.to_csv("nhanes.csv.gz", index=False, compression="gzip")

import (
	"compress/gzip"
	"fmt"
	"math"
	"os"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/dstream/formula"
	"github.com/kshedden/goglm"
	"github.com/kshedden/statmodel"
)

func getData() dstream.Dstream {

	fid, err := os.Open("nhanes.csv.gz")
	if err != nil {
		panic(err)
	}
	defer fid.Close()
	gid, err := gzip.NewReader(fid)
	if err != nil {
		panic(err)
	}
	defer gid.Close()

	keepfloat := []string{"RIAGENDR", "RIDAGEYR", "BPXSY1"}
	keepstring := []string{"RIDRETH1"}

	tc := &dstream.CSVTypeConf{
		Float64: keepfloat,
		String:  keepstring,
	}

	dst := dstream.FromCSV(gid).TypeConf(tc).SetChunkSize(100).HasHeader().Done()
	dsc := dstream.MemCopy(dst)

	dsc.Reset()
	dsc.Next()

	return dsc
}

func model1() {

	msg := `
Linear regression (ordinary least squares) for systolic blood pressure,
using two predictor variables: gender (RIAGENDR) and age (RIDAGEYR).
Gender is treated as a quantitative variable and is coded as 1 for
males and 2 for females.
`

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR"

	f1 := formula.New(fml, dp).Keep("BPXSY1").Done()
	f2 := dstream.MemCopy(f1)
	f3 := dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f3, "BPXSY1").Family(fam).Done()
	rslt := glm.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model2() {

	msg := `
Linear regression (ordinary least squares) for systolic blood pressure,
including ethnicity as a categorical covariate, using level 5 (other
race/multiracial) as the reference category.
`

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp).RefLevels(reflev).Keep("BPXSY1").Done()
	f2 := dstream.MemCopy(f1)
	f2 = dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).Done()
	rslt := glm.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model3() {

	msg := `
Linear regression (ordinary least squares) for systolic blood pressure,
including gender, age, ethnicity, and the interaction between gender
and age as covariates.  Ethnicity is a categorical covariate with level
5 (other race/multiracial) as the reference category.
`

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1 + RIAGENDR * RIDAGEYR"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp).RefLevels(reflev).Keep("BPXSY1").Done()
	f2 := dstream.MemCopy(f1)
	f2 = dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).Done()
	rslt := glm.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model4() {

	msg := `
Regularized least squares regression (Lasso regression) for systolic
blood pressure, using equal penalty weights for all covariates and
zero penalty for the intercept.
`

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp).Keep("BPXSY1").RefLevels(reflev).Done()
	f1 = dstream.DropNA(f1)
	f2 := dstream.MemCopy(f1)

	wt := 0.01
	l1wgt := []float64{0}
	for i := 0; i < 6; i++ {
		l1wgt = append(l1wgt, wt)
	}

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).L1Weight(l1wgt).CovariateScale(statmodel.L2Norm).Done()

	rslt := glm.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model5() {

	msg := `
Linear regression with systolic blood pressure as the outcome,
using a square root transform in the formula.
`

	dp := getData()

	fml := "1 + RIAGENDR + sqrt(RIDAGEYR) + RIDRETH1"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	funcs := make(map[string]formula.Func)
	funcs["sqrt"] = func(na string, x []float64) *formula.ColSet {
		y := make([]float64, len(x))
		for i, v := range x {
			y[i] = v * v
		}
		return &formula.ColSet{
			Names: []string{na},
			Data:  []interface{}{y},
		}
	}

	f1 := formula.New(fml, dp).Keep("BPXSY1").RefLevels(reflev).Funcs(funcs).Done()
	f2 := dstream.MemCopy(f1)
	f2 = dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).Done()

	rslt := glm.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

// Create a binary indicator of high systolic blood pressure.
func hbp(v map[string]interface{}, x interface{}) {
	z := x.([]float64)
	bp := v["BPXSY1"].([]float64)
	for i := range bp {
		if bp[i] >= 130 {
			z[i] = 1
		} else {
			z[i] = 0
		}
	}
}

func model6() {

	msg := `
Logistic regression using high blood pressure status (binary) as
the dependent variable, and gender and age as predictors.
`

	dp := getData()

	dp.Reset()
	dp = dstream.Generate(dp, "BP", hbp, dstream.Float64)
	dp = dstream.MemCopy(dp)

	fml := "1 + RIAGENDR + RIDAGEYR"

	f1 := formula.New(fml, dp).Keep("BP").Done()
	f2 := dstream.MemCopy(f1)
	f3 := dstream.DropNA(f2)

	fam := goglm.NewFamily("binomial")
	glm := goglm.NewGLM(f3, "BP").Family(fam).CovariateScale(statmodel.L2Norm).Done()
	rslt := glm.Fit()

	smry := rslt.Summary()
	fmt.Printf(msg + "\n")
	fmt.Printf(smry.String() + "\n\n")

	smry = smry.SetScale(math.Exp, "Parameters are shown as odds ratios")

	fmt.Printf(smry.String() + "\n\n")
}

func model7() {

	msg := `
Elastic net penalized logistic regression for high blood pressure
status, with L1 and L2 penalties.  Age and gender are the predictor
variables.
`

	dp := getData()

	dp.Reset()
	dp = dstream.Generate(dp, "BP", hbp, dstream.Float64)
	dp = dstream.MemCopy(dp)

	fml := "1 + RIAGENDR + RIDAGEYR"

	f1 := formula.New(fml, dp).Keep("BP").Done()
	f2 := dstream.MemCopy(f1)
	f3 := dstream.DropNA(f2)

	l1wgt := []float64{0, 1, 0}
	l2wgt := []float64{0.01, 0.01, 0.01}

	fam := goglm.NewFamily("binomial")
	glm := goglm.NewGLM(f3, "BP").Family(fam).L1Weight(l1wgt).L2Weight(l2wgt).CovariateScale(statmodel.L2Norm).Done()
	rslt := glm.Fit()
	smry := rslt.Summary()

	fmt.Printf(msg + "\n")
	fmt.Printf(smry.String() + "\n\n")
}

func main() {
	model1()
	model2()
	model3()
	model4()
	model5()
	model6()
	model7()
}
