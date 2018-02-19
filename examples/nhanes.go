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

	dst := dstream.FromCSV(gid).SetFloatVars(keepfloat).SetStringVars(keepstring).SetChunkSize(100).HasHeader().Done()
	dsc := dstream.MemCopy(dst)

	dsc.Reset()
	dsc.Next()

	return dsc
}

func model1() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR"

	f1 := formula.New(fml, dp).Keep([]string{"BPXSY1"}).Done()
	f2 := dstream.MemCopy(f1)
	f3 := dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f3, "BPXSY1").Family(fam).Done()
	rslt := glm.Fit()
	print(rslt.Summary() + "\n\n")
}

func model2() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp).RefLevels(reflev).Keep([]string{"BPXSY1"}).Done()
	f2 := dstream.MemCopy(f1)
	f2 = dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).Done()
	rslt := glm.Fit()
	print(rslt.Summary() + "\n\n")
}

func model3() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1 + RIAGENDR * RIDAGEYR"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp).RefLevels(reflev).Keep([]string{"BPXSY1"}).Done()
	f2 := dstream.MemCopy(f1)
	f2 = dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).Done()
	rslt := glm.Fit()
	print(rslt.Summary() + "\n\n")
}

func model4() {

	dp := getData()

	fml := "1 + RIAGENDR + RIDAGEYR + RIDRETH1"
	reflev := map[string]string{"RIDRETH1": "5.0"}

	f1 := formula.New(fml, dp).Keep([]string{"BPXSY1"}).RefLevels(reflev).Done()
	f1 = dstream.DropNA(f1)
	f2 := dstream.MemCopy(f1)

	wt := 0.01
	l1wgt := []float64{0}
	for i := 0; i < 6; i++ {
		l1wgt = append(l1wgt, wt)
	}

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).L1Weight(l1wgt).Norm().Done()

	rslt := glm.Fit()
	print(rslt.Summary() + "\n\n")
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
			Data:  []interface{}{y},
		}
	}

	f1 := formula.New(fml, dp).Keep([]string{"BPXSY1"}).RefLevels(reflev).Funcs(funcs).Done()
	f2 := dstream.MemCopy(f1)
	f2 = dstream.DropNA(f2)

	fam := goglm.NewFamily("gaussian")
	glm := goglm.NewGLM(f2, "BPXSY1").Family(fam).Done()

	rslt := glm.Fit()
	print(rslt.Summary() + "\n\n")
}

func model6() {

	dp := getData()

	di := func(v map[string]interface{}, x interface{}) {
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

	dp.Reset()
	dp = dstream.Apply(dp, "BP", di, "float64")
	dp = dstream.MemCopy(dp)

	fml := "1 + RIAGENDR + RIDAGEYR"

	f1 := formula.New(fml, dp).Keep([]string{"BP"}).Done()
	f2 := dstream.MemCopy(f1)
	f3 := dstream.DropNA(f2)

	l1wgt := []float64{0.1, 0.1, 0.1}
	l2wgt := []float64{1, 1, 1}

	fam := goglm.NewFamily("binomial")
	glm := goglm.NewGLM(f3, "BP").Family(fam).L1Weight(l1wgt).L2Weight(l2wgt).Norm().Done()
	rslt := glm.Fit()
	print(rslt.Summary() + "\n\n")
}

func main() {
	model1()
	model2()
	model3()
	model4()
	model5()
	model6()
}
