clear
clear mata
set mem 30m

use "figae_fig2_data.dta", clear
spearman  cslope0 pslope
spearman  caveragecontribution paveragecontribution

use "figae_beliefs&contributions.dta", clear

set matsize 800

* regressions

gen idgroup=10*idsess+group


tsset idsubj period 
*** regression for Table 1

* Model 1
reg  belief  period  if period >1, r cl( idsess )
* Model 2
reg  belief period l.otherscontrib   l.belief    , r cl( idsess )
* Model 3
reg  belief l.otherscontrib   l.belief    , r cl( idsess )
* the nest test is used in section E
test l.belief + l.otherscontrib ==1

* Chow test for model 2
gen half1=2-half
gen half2=half-1
gen l_otherscontrib=l.otherscontrib
gen l_otherscontrib1=l_otherscontrib*half1
gen l_otherscontrib2=l_otherscontrib*half2
gen l_belief=l.belief    
gen l_belief1=l_belief*half1
gen l_belief2=l_belief*half2
gen period1=period*half1
gen period2=period*half2
reg  belief half1 period1 l_otherscontrib1 l_belief1 half2 period2 l_otherscontrib2 l_belief2 , noconst r cl( idsess )
test _b[period1] = _b[period2]
test _b[l_otherscontrib1 ] = _b[l_otherscontrib2], accum
test _b[l_belief1 ] = _b[l_belief2], accum
test _b[half1 ] = _b[half2 ], accum

* period coefficient different in Models 1 and 2
reg  belief  period  if period >1
est store regT1R1
reg  belief period l.otherscontrib   l.belief   
est store regT1R2
reg  belief l.otherscontrib   l.belief    
est store regT1R3
suest regT1R1 regT1R2, cl( idsess )
test [regT1R1 _mean]period = [regT1R2_mean]period = 0


** FN 13
xtabond  belief l(1/3).otherscontrib, lags(1) r
reg  belief l.otherscontrib   l.belief   if period ==2 , r cl( idsess )


** Table 2
* Model 1
reg contribution  period , r cl( idsess )
* Model 2
reg contribution  period predictedcontribution belief  , r cl( idsess )
* Model 3
reg contribution  predictedcontribution belief  , r cl( idsess )
vif
spearman predictedcontribution belief  

* Model 4a
reg contribution   predictedcontribution belief   if idtyp<4 , r cl( idsess )
test predictedcontribution+ belief  =1
* Model 4b
reg contribution   predictedcontribution belief   if idtyp<4 & period <=5, r cl( idsess )
test predictedcontribution+ belief  =1
test predictedcontribution=belief   
* Model 4c
reg contribution   predictedcontribution belief   if idtyp<4 & period >5, r cl( idsess )
test predictedcontribution+ belief  =1
test predictedcontribution=belief   
* Comparing Model 4b and Moden 4c
xi:reg contribution   i.half2*predictedcontribution i.half2*belief   if idtyp<4 , r cl( idsess )

* coefficient of predictedcontribution higher in Model 4a than in Model 3
reg contribution  predictedcontribution belief   
est store regT2R3
reg contribution   predictedcontribution belief   if idtyp<4 
est store regT2R4a
suest regT2R3 regT2R4a, cl( idsess )
test [regT2R3_mean]predictedcontribution= [regT2R4a_mean]predictedcontribution

* robustness tests
reg  contribution period predictedcontribution belief    , r cl( idsess )
areg  contribution period predictedcontribution  belief    , a( idsubj ) r cl( idsess)
xtreg  contribution period predictedcontribution   belief    , i( idsubj ) 
tobit  contribution period predictedcontribution   belief    , ul(20) ll(0) 



* SAMAB = simple adaptive model with actual beliefs 
* Models differ in calculation of belief and of conditional cooperation

* BN Belief naiv = av contribution of others in previous period
* BA Belief actual = estimated according to Model 3 in Table 1

* ACC actual conditional cooperation
* PCC perfect conditional cooperation
* iCC identical  conditional cooperation

gen SAMABbelief = .
gen SAMABcc = .
gen SAMABcontribution = .
 
gen x=0 
gen y=0
gen v=0



replace SAMABbelief = belief
replace x=SAMABbelief 
replace y=0
replace y=b0 if x==0
replace y=b1 if x==1
replace y=b2 if x==2
replace y=b3 if x==3
replace y=b4 if x==4
replace y=b5 if x==5
replace y=b6 if x==6
replace y=b7 if x==7
replace y=b8 if x==8
replace y=b9 if x==9
replace y=b10 if x==10
replace y=b11 if x==11
replace y=b12 if x==12
replace y=b13 if x==13
replace y=b14 if x==14
replace y=b15 if x==15
replace y=b16 if x==16
replace y=b17 if x==17
replace y=b18 if x==18
replace y=b19 if x==19
replace y=b20 if x==20
replace SAMABcc = y
replace SAMABcontribution = SAMABcc 



* SIM = general simulation


gen SIMbelief = .
gen SIMacc = .
gen SIMhcc = .
gen SIMcontribution = . 
gen SIMsumgrp = .
gen SIMavothers = .


foreach beliefvariant in "BN" "BA"  {
foreach ccvariant in "aCC" "iCC" "pCC"  {

replace  SIMbelief = belief



forvalues i=1(1)10 {
tsset idsubj period 
replace SIMbelief = round( .415*l.SIMavothers + .569* l.SIMbelief +0.118)  if period>1 & "`beliefvariant'" =="BA"
replace SIMbelief = round(  l.SIMavothers  )  if period>1 &  "`beliefvariant'" == "BN"
replace x=SIMbelief 
replace y=0
replace y=b0 if x==0
replace y=b1 if x==1
replace y=b2 if x==2
replace y=b3 if x==3
replace y=b4 if x==4
replace y=b5 if x==5
replace y=b6 if x==6
replace y=b7 if x==7
replace y=b8 if x==8
replace y=b9 if x==9
replace y=b10 if x==10
replace y=b11 if x==11
replace y=b12 if x==12
replace y=b13 if x==13
replace y=b14 if x==14
replace y=b15 if x==15
replace y=b16 if x==16
replace y=b17 if x==17
replace y=b18 if x==18
replace y=b19 if x==19
replace y=b20 if x==20
replace SIMacc = y  

replace SIMcontribution = round( .242* SIMacc + .666*SIMbelief -0.473  )      if "`ccvariant'"=="aCC"
replace SIMcontribution = round( .242*(0.425*SIMbelief + 0.956)+ .666*SIMbelief -0.473   )      if "`ccvariant'"=="iCC"
replace SIMcontribution = round( SIMbelief   )      if "`ccvariant'"=="pCC"

replace SIMcontribution = contribution if period ==1

sort idgroup period
drop v
egen v = sum(SIMcontribution), by(idgroup period)
replace SIMsumgrp = v
replace SIMavothers = round( (SIMsumgrp - SIMcontribution )/3 ) 
}
gen SIM`beliefvariant'`ccvariant' = SIMcontribution 
gen SIM`beliefvariant'`ccvariant'_diff = SIM`beliefvariant'`ccvariant' -  contribution 
gen SIM`beliefvariant'`ccvariant'_absdiff = abs( SIM`beliefvariant'`ccvariant' -  contribution )
gen SIM`beliefvariant'`ccvariant'_sqrdiff = abs( SIM`beliefvariant'`ccvariant' -  contribution )^2
gen SIMBEL`beliefvariant'`ccvariant' = SIMbelief 

}
}


* assess the quality of the simulation 
tsset idsubj period 

foreach beliefvariant in "BN" "BA"  {
  foreach ccvariant in "aCC" "iCC" "pCC" {
    sum SIM`beliefvariant'`ccvariant'
    spearman contribution  SIM`beliefvariant'`ccvariant'
    correl contribution  SIM`beliefvariant'`ccvariant'
    reg contribution SIM`beliefvariant'`ccvariant', r cl ( idsess) 
    test SIM`beliefvariant'`ccvariant'==1
  }
}




foreach beliefvariant in "BN" "BA" {
  foreach ccvariant in "aCC" "iCC" "pCC" {
    sum SIM`beliefvariant'`ccvariant'
    spearman belief SIMBEL`beliefvariant'`ccvariant'
    correl belief SIMBEL`beliefvariant'`ccvariant'
    reg belief SIMBEL`beliefvariant'`ccvariant', r cl ( idsess) 
    test SIMBEL`beliefvariant'`ccvariant'==1
  }
}

* data for Figure 2B
sort idsubj 
by idsubj: sum contribution
by idsubj: reg contribution belief  
by idsubj: reg contribution belief,noconst  



save ".\figae_simulations.dta", replace


use ".\condcoopsimulations.dta", clear
sort period 
collapse (mean)  ///
mn_contribution = contribution ///
mn_SIMBNaCC = SIMBNaCC ///
mn_SIMBNiCC = SIMBNiCC ///
mn_SIMBNpCC = SIMBNpCC ///
mn_SIMBAaCC = SIMBAaCC ///
mn_SIMBAiCC = SIMBAiCC ///
mn_SIMBApCC = SIMBApCC ///
(count) ///
 count = contribution ///
, by( period )
outsheet using ".\simulationres.txt", replace

use ".\condcoopsimulations.dta", clear
sort idsess period 
collapse (mean)  ///
mn_contribution = contribution ///
mn_SIMBNaCC = SIMBNaCC ///
mn_SIMBNiCC = SIMBNiCC ///
mn_SIMBNpCC = SIMBNpCC ///
mn_SIMBAaCC = SIMBAaCC ///
mn_SIMBAiCC = SIMBAiCC ///
mn_SIMBApCC = SIMBApCC ///
(count) ///
 count = contribution ///
, by( idsess period )
outsheet using ".\simulationsessionres.txt", replace


