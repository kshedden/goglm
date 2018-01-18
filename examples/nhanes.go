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
	"os"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/dstream/formula"
	"github.com/kshedden/goglm"
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

	dst := dstream.FromCSV(gid)
	dst.SetFloatVars(keepfloat)
	dst.SetStringVars(keepstring)
	dst.SetChunkSize(100)
	dst.Init(true)

	dsc := dstream.MemCopy(dst)

	dsc.Reset()
	dsc.Next()

	return dsc
}

func model1() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp, reflev, nil, nil)
	f2 := f1.ParseAll([]string{"BPXSY1"})
	f3 := dstream.DropNA(f2)
	xn := []string{"icept", "RIAGENDR", "RIDAGEYR"}
	f4 := dstream.NewReg(f3, "BPXSY1", xn, "", "")

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(fam, f4)
	rslt := glm.Fit()
	print(rslt.Summary())
}

func model2() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp, reflev, nil, nil)
	f2 := f1.ParseAll([]string{"BPXSY1"})
	f3 := dstream.DropNA(f2)
	f4 := dstream.NewReg(f3, "BPXSY1", nil, "", "")

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(fam, f4)
	rslt := glm.Fit()
	print(rslt.Summary())
}

func model3() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1 + RIAGENDR * RIDAGEYR"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp, reflev, nil, nil)
	f2 := f1.ParseAll([]string{"BPXSY1"})
	f3 := dstream.DropNA(f2)
	f4 := dstream.NewReg(f3, "BPXSY1", nil, "", "")

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(fam, f4)
	rslt := glm.Fit()
	print(rslt.Summary())
}

func model4() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp, reflev, nil, nil)
	f2 := f1.ParseAll([]string{"BPXSY1"})
	f3 := dstream.DropNA(f2)
	f4 := dstream.NewReg(f3, "BPXSY1", nil, "", "")

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(fam, f4)

	wt := 0.01
	glm.L1wgt = make([]float64, f4.NumCov())
	for i := 1; i < f4.NumCov(); i++ {
		glm.L1wgt[i] = wt
	}

	rslt := glm.Fit()
	print(rslt.Summary())
}

func model5() {

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
			Data:  [][]float64{y},
		}
	}

	f1 := formula.New(fml, dp, reflev, nil, funcs)
	f2 := f1.ParseAll([]string{"BPXSY1"})
	f3 := dstream.DropNA(f2)
	f4 := dstream.NewReg(f3, "BPXSY1", nil, "", "")

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(fam, f4)

	rslt := glm.Fit()
	print(rslt.Summary())
}

func main() {
	model1()
	model2()
	model3()
	model4()
	model5()
}
