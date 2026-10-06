Rz angle library -- CORRECTED (2026-10-05)
==========================================

Same format as rz_library_sohan.zip. What changed and why:

1. T counts were understated (ERRATA E1). The old column t_exact was the
   denominator exponent k of the ancilla unitary V (upstream sde()), documented
   as "exact T count". Exact synthesis of every gadget shows the true T count of
   one attempt is 2k-2 or 2k. All expected costs were therefore too low.
2. Cheapest dressing (ERRATA E4). For each gadget, the ancilla unitary can be
   dressed as V' = diag(1, w^a) V diag(1, w^b) at no cost. The same-angle dressing
   (a+b = 0 mod 8) is sometimes cheaper, and the odd-shift dressing (angle + pi/4,
   or equivalently - pi/4) is sometimes cheaper than "same angle + one T". Each
   gadget now has TWO rows:
     source = brs_maxL7 / brs_L89                 cheapest same-angle dressing
     source = brs_maxL7_oddshift / brs_L89_oddshift  cheapest odd-shift dressing
   build_rz_library.dress() then gives every dressed point its true price. z, y
   are the ring elements of the DRESSED ancilla unitary.
3. Catalysis removed. rz_catalyst_angles.csv (k pi/12 at a declared flat 5 T,
   never verified) and the cat1 / cat3 libraries (Gidney catalysis, multi-level)
   are not included. Note: the old cat0 library also contained the catalyst
   angles. The corrected cat0 contains RUS gadgets and k pi/4 anchors only.

Net effect: library costs rise by 1.65x at the median (10th-90th percentile:
1.45-1.74x over the 27,474 non-Clifford angles shared with the old file).

Every row is an explicit circuit, independently verified: the dressed V' is
synthesised exactly and the CNOT-V'-CNOT-measure circuit is simulated. Angle,
success probability, failure correction class and T count are read off the
circuit, not from a formula. The corrected cat0 library agrees with the
reference harness's pricing of the same gadgets to 7e-14 at every common angle.

Convention: Rz(theta) = diag(e^{-i theta/2}, e^{+i theta/2}), as before. Costs
are expected T counts with retries included. Generic failure corrections are
priced at 18 T (E_T_kn18 and the library) or 50 T (E_T_rs50), as before.
theta_f_pi follows the old file's convention.

rz_rus_gadgets.csv   (4,580 rows = 2,290 gadgets x 2 dressings; same columns)
rz_library_cat0_kn18.csv   (27,499 rows; same columns; is_catalytic always 0)
build_rz_library.py   (unchanged functions; __main__ now builds cat0 only, with
                       no catalyst file)

Regenerate the library with:
    S, C = build_rz_library.build("rz_rus_gadgets.csv", None, cat_levels=0)
