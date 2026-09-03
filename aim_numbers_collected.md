# AIM Scratch and Tuning Results

Source AIM repository: `/mnt/weka/asaribekyan/iot/logs/rome_new`

Included runs: created on or after `2026-06-01` with experiment labels matching `scratch_{img_size}_{epochs}_{lr}` or `tune_{dataset}_{img_size}_{epochs}_{lr}`.

Values are localization error in meters. Lower is better.

Metric definitions:

- `hard_test`: final single test-set dot from `0_mse_meters`, `subset=test`.
- `hard_val`: best/minimum hard validation value from `0_mse_meters`, `subset=val`.
- `medium_test`: final single test-set dot from `1_mse_meters`, `subset=test`.
- `medium_val`: best/minimum medium validation value from `1_mse_meters`, `subset=val`.
- `medium_val_at_hard_best`: medium validation value at the same epoch index where hard validation is best.

Notes:

- Row keys and dataset groups are parsed from AIM experiment labels. The logged config has `map_size=448` and often `optimizer.lr=0.0003`, while the label carries the requested comparison key such as `1000_80_6e-5`.
- The automatic test dot is produced after training with the final in-memory model, not by reloading the validation-best checkpoints.
- If a validation minimum ties exactly, the first occurrence is used for `medium_val_at_hard_best`.

<table>
  <thead>
    <tr><th rowspan="2">setup</th><th colspan="5">scratch</th><th colspan="5">bp-const</th><th colspan="5">bp-unconst</th><th colspan="5">c-const</th><th colspan="5">c-const-norm</th><th colspan="5">c-unconst</th></tr>
    <tr><th>hard_test</th><th>hard_val</th><th>medium_test</th><th>medium_val</th><th>medium_val_at_hard_best</th><th>hard_test</th><th>hard_val</th><th>medium_test</th><th>medium_val</th><th>medium_val_at_hard_best</th><th>hard_test</th><th>hard_val</th><th>medium_test</th><th>medium_val</th><th>medium_val_at_hard_best</th><th>hard_test</th><th>hard_val</th><th>medium_test</th><th>medium_val</th><th>medium_val_at_hard_best</th><th>hard_test</th><th>hard_val</th><th>medium_test</th><th>medium_val</th><th>medium_val_at_hard_best</th><th>hard_test</th><th>hard_val</th><th>medium_test</th><th>medium_val</th><th>medium_val_at_hard_best</th></tr>
  </thead>
  <tbody>
    <tr><td>1000_10_6e-5</td><td>747.10</td><td>682.54</td><td>719.38</td><td>719.25</td><td>736.10</td><td>245.64</td><td>319.96</td><td>182.38</td><td>127.98</td><td>194.03</td><td>306.81</td><td>299.38</td><td>206.46</td><td>125.72</td><td>168.53</td><td>378.45</td><td>295.36</td><td>200.90</td><td>105.00</td><td>209.71</td><td>227.62</td><td>305.64</td><td>241.03</td><td>153.99</td><td>259.41</td><td>305.90</td><td>292.25</td><td>195.26</td><td>130.07</td><td>204.01</td></tr>
    <tr><td>1000_10_2e-4</td><td>448.84</td><td>477.44</td><td>548.23</td><td>531.91</td><td>531.91</td><td>222.50</td><td>307.25</td><td>126.12</td><td>126.65</td><td>176.36</td><td>359.97</td><td>325.61</td><td>173.24</td><td>107.65</td><td>168.56</td><td>372.29</td><td>299.62</td><td>160.77</td><td>102.67</td><td>179.88</td><td>230.22</td><td>306.05</td><td>262.86</td><td>127.19</td><td>220.72</td><td>302.28</td><td>293.68</td><td>148.77</td><td>82.31</td><td>197.60</td></tr>
    <tr><td>1000_10_6e-4</td><td>801.25</td><td>638.92</td><td>721.67</td><td>586.39</td><td>586.39</td><td>256.98</td><td>331.93</td><td>122.82</td><td>108.38</td><td>145.79</td><td>337.36</td><td>308.24</td><td>139.41</td><td>85.82</td><td>119.88</td><td>352.22</td><td>305.46</td><td>165.56</td><td>88.17</td><td>144.75</td><td>232.48</td><td>302.30</td><td>156.50</td><td>105.63</td><td>220.81</td><td>280.65</td><td>309.34</td><td>147.30</td><td>94.75</td><td>144.34</td></tr>
    <tr><td>1000_20_6e-5</td><td>744.99</td><td>687.06</td><td>752.27</td><td>637.63</td><td>751.51</td><td>220.10</td><td>297.53</td><td>163.26</td><td>123.69</td><td>142.67</td><td>297.79</td><td>297.19</td><td>169.89</td><td>110.17</td><td>167.98</td><td>378.45</td><td>295.36</td><td>200.90</td><td>96.96</td><td>209.71</td><td>227.62</td><td>305.64</td><td>241.03</td><td>146.97</td><td>259.41</td><td>305.90</td><td>292.25</td><td>195.26</td><td>109.76</td><td>204.01</td></tr>
    <tr><td>1000_20_2e-4</td><td>563.16</td><td>465.52</td><td>546.81</td><td>538.42</td><td>538.42</td><td>236.18</td><td>300.21</td><td>119.82</td><td>69.91</td><td>110.26</td><td>364.17</td><td>319.65</td><td>171.14</td><td>86.74</td><td>86.97</td><td>372.29</td><td>299.62</td><td>160.77</td><td>75.39</td><td>179.88</td><td>230.22</td><td>306.05</td><td>262.86</td><td>102.12</td><td>220.72</td><td>302.13</td><td>293.96</td><td>146.81</td><td>78.58</td><td>197.66</td></tr>
    <tr><td>1000_20_6e-4</td><td>368.37</td><td>420.38</td><td>491.35</td><td>496.04</td><td>496.04</td><td>272.89</td><td>320.44</td><td>143.48</td><td>64.95</td><td>182.99</td><td>320.68</td><td>324.55</td><td>166.09</td><td>75.31</td><td>86.57</td><td>352.22</td><td>313.22</td><td>165.56</td><td>76.40</td><td>135.20</td><td>201.79</td><td>309.62</td><td>274.40</td><td>80.62</td><td>218.82</td><td>273.26</td><td>291.16</td><td>145.61</td><td>78.03</td><td>138.96</td></tr>
    <tr><td>1000_40_6e-5</td><td>301.62</td><td>315.81</td><td>156.41</td><td>123.53</td><td>152.73</td><td>217.89</td><td>320.01</td><td>161.25</td><td>81.17</td><td>154.71</td><td>334.95</td><td>297.36</td><td>232.38</td><td>87.87</td><td>164.93</td><td>378.45</td><td>295.36</td><td>200.90</td><td>72.13</td><td>209.71</td><td>245.37</td><td>281.80</td><td>166.68</td><td>104.77</td><td>163.66</td><td>305.90</td><td>292.25</td><td>195.26</td><td>92.78</td><td>204.01</td></tr>
    <tr><td>1000_40_2e-4</td><td>268.81</td><td>351.04</td><td>242.12</td><td>197.84</td><td>197.84</td><td>239.36</td><td>309.40</td><td>159.73</td><td>69.88</td><td>187.96</td><td>365.52</td><td>321.38</td><td>171.00</td><td>79.75</td><td>86.91</td><td>372.29</td><td>299.62</td><td>160.77</td><td>64.92</td><td>179.88</td><td>212.96</td><td>305.50</td><td>165.26</td><td>70.24</td><td>122.92</td><td>302.13</td><td>293.96</td><td>146.81</td><td>69.09</td><td>197.66</td></tr>
    <tr><td>1000_40_6e-4</td><td>346.14</td><td>430.03</td><td>344.70</td><td>353.19</td><td>384.06</td><td>251.65</td><td>290.34</td><td>160.41</td><td>68.78</td><td>161.52</td><td>322.55</td><td>310.09</td><td>214.76</td><td>75.65</td><td>90.17</td><td>352.22</td><td>304.85</td><td>165.56</td><td>64.81</td><td>161.05</td><td>201.79</td><td>309.62</td><td>274.40</td><td>80.21</td><td>218.82</td><td>276.19</td><td>290.96</td><td>151.66</td><td>71.75</td><td>163.77</td></tr>
    <tr><td>1000_80_6e-5</td><td>289.52</td><td>353.48</td><td>333.77</td><td>231.62</td><td>231.62</td><td>227.62</td><td>330.05</td><td>174.93</td><td>61.20</td><td>123.61</td><td>334.95</td><td>297.32</td><td>232.38</td><td>70.15</td><td>167.46</td><td>378.45</td><td>295.36</td><td>200.90</td><td>60.03</td><td>209.71</td><td>212.96</td><td>277.89</td><td>178.94</td><td>82.15</td><td>149.07</td><td>305.90</td><td>292.25</td><td>195.26</td><td>71.28</td><td>204.01</td></tr>
    <tr><td>1000_80_2e-4</td><td>274.31</td><td>284.84</td><td>186.86</td><td>121.73</td><td>207.98</td><td>232.88</td><td>322.74</td><td>126.88</td><td>58.68</td><td>161.17</td><td>365.51</td><td>318.60</td><td>171.02</td><td>62.91</td><td>89.29</td><td>372.31</td><td>299.62</td><td>160.77</td><td>50.45</td><td>179.88</td><td>211.34</td><td>302.73</td><td>172.64</td><td>61.36</td><td>126.48</td><td>302.13</td><td>293.96</td><td>146.81</td><td>56.51</td><td>197.66</td></tr>
    <tr><td>1000_80_6e-4</td><td>277.30</td><td>309.94</td><td>209.72</td><td>137.95</td><td>200.52</td><td>243.28</td><td>324.70</td><td>144.50</td><td>51.15</td><td>85.90</td><td>295.65</td><td>309.15</td><td>153.00</td><td>57.14</td><td>112.04</td><td>352.22</td><td>313.72</td><td>165.56</td><td>43.68</td><td>196.93</td><td>202.42</td><td>309.54</td><td>198.09</td><td>44.32</td><td>220.64</td><td>277.83</td><td>280.19</td><td>147.28</td><td>54.43</td><td>152.08</td></tr>
    <tr><td>1000_100_6e-5</td><td>269.54</td><td>301.98</td><td>185.90</td><td>123.51</td><td>144.82</td><td>224.93</td><td>319.53</td><td>170.25</td><td>57.35</td><td>164.53</td><td>334.95</td><td>297.32</td><td>232.38</td><td>67.37</td><td>167.46</td><td>378.45</td><td>295.36</td><td>200.90</td><td>60.36</td><td>209.71</td><td>211.35</td><td>281.26</td><td>178.32</td><td>75.42</td><td>150.19</td><td>305.90</td><td>292.25</td><td>195.26</td><td>51.70</td><td>204.01</td></tr>
    <tr><td>1000_100_2e-4</td><td>258.55</td><td>319.11</td><td>190.30</td><td>134.30</td><td>155.61</td><td>233.17</td><td>317.01</td><td>147.84</td><td>46.16</td><td>115.28</td><td>365.51</td><td>318.28</td><td>171.09</td><td>48.08</td><td>89.29</td><td>372.28</td><td>299.62</td><td>160.77</td><td>46.28</td><td>179.88</td><td>213.95</td><td>300.44</td><td>174.51</td><td>57.98</td><td>131.51</td><td>302.13</td><td>293.96</td><td>146.81</td><td>49.75</td><td>197.66</td></tr>
    <tr><td>1000_100_6e-4</td><td>300.18</td><td>344.77</td><td>278.51</td><td>199.97</td><td>235.20</td><td>221.45</td><td>315.01</td><td>168.64</td><td>48.46</td><td>100.55</td><td>322.55</td><td>311.78</td><td>214.76</td><td>51.43</td><td>117.76</td><td>352.22</td><td>313.72</td><td>165.56</td><td>49.29</td><td>196.93</td><td>230.84</td><td>283.25</td><td>140.94</td><td>43.94</td><td>143.28</td><td>292.20</td><td>300.23</td><td>138.43</td><td>50.28</td><td>101.43</td></tr>
    <tr><td>1000_120_6e-5</td><td>233.75</td><td>313.26</td><td>178.04</td><td>116.03</td><td>216.60</td><td>224.03</td><td>325.13</td><td>171.88</td><td>50.45</td><td>155.27</td><td>334.95</td><td>297.29</td><td>232.38</td><td>68.32</td><td>167.42</td><td>378.45</td><td>295.36</td><td>200.90</td><td>60.88</td><td>209.71</td><td>214.07</td><td>282.26</td><td>181.13</td><td>69.63</td><td>150.37</td><td>305.90</td><td>292.25</td><td>195.26</td><td>47.02</td><td>204.01</td></tr>
    <tr><td>1000_120_2e-4</td><td>262.09</td><td>321.09</td><td>218.11</td><td>97.84</td><td>189.45</td><td>202.97</td><td>312.43</td><td>97.38</td><td>51.64</td><td>104.73</td><td>366.26</td><td>318.32</td><td>170.80</td><td>55.55</td><td>89.15</td><td>372.28</td><td>299.62</td><td>160.77</td><td>45.85</td><td>179.88</td><td>213.32</td><td>299.07</td><td>170.98</td><td>53.45</td><td>135.30</td><td>302.13</td><td>293.96</td><td>146.81</td><td>49.79</td><td>197.66</td></tr>
    <tr><td>1000_120_6e-4</td><td>342.09</td><td>364.19</td><td>313.27</td><td>260.25</td><td>320.80</td><td>202.41</td><td>310.91</td><td>119.34</td><td>52.14</td><td>111.63</td><td>321.34</td><td>290.91</td><td>144.22</td><td>51.96</td><td>95.45</td><td>352.22</td><td>313.58</td><td>165.56</td><td>46.28</td><td>147.55</td><td>201.79</td><td>312.48</td><td>274.40</td><td>45.77</td><td>220.64</td><td>270.44</td><td>267.50</td><td>106.63</td><td>43.87</td><td>104.37</td></tr>
  </tbody>
</table>

## Pretraining Results

These rows are the pretraining experiments used as sources for the tuning groups above. The same five metrics are reported.

All five runs were found exactly once in AIM. Each has 300 hard/medium validation points and one hard/medium test point. `bp-unconst-str-2e-4` is the only one of these five whose logged config has a non-null `ckpt_path`; it loaded `/mnt/weka/asaribekyan/iot/outputs/2026-06-03_03-08-31.283278/checkpoints/hard/last.ckpt`.

| experiment | hard_test | hard_val | medium_test | medium_val | medium_val_at_hard_best |
|---|---:|---:|---:|---:|---:|
| `bp-const-str-2e-4` | 138.47 | 182.63 | 120.65 | 51.43 | 104.68 |
| `bp-unconst-str-2e-4` | 187.16 | 183.85 | 60.43 | 42.08 | 94.88 |
| `c-const-str-2e-4` | 264.17 | 227.49 | 205.48 | 79.84 | 213.15 |
| `c-const-str-2e-4-normalized` | 273.64 | 230.96 | 121.89 | 82.36 | 120.25 |
| `c-unconst-str-2e-4` | 188.10 | 151.89 | 123.33 | 56.22 | 85.07 |

### Pretraining Run Mapping

| experiment | hash | created_at | hard_best_epoch_index | medium_best_epoch_index |
|---|---|---:|---:|---:|
| `bp-const-str-2e-4` | `2b24ac2bfa5340119db2a37a` | 2026-06-05 05:26:29 | 229 | 173 |
| `bp-unconst-str-2e-4` | `1da45c0c754c4a689f155905` | 2026-06-02 23:08:51 | 44 | 195 |
| `c-const-str-2e-4` | `ef9525373f26469bb0d5ecd7` | 2026-06-15 12:26:21 | 79 | 113 |
| `c-const-str-2e-4-normalized` | `e18487a8c0794f67afcf326e` | 2026-06-16 06:04:36 | 94 | 222 |
| `c-unconst-str-2e-4` | `b51300c4e88849e6be40f356` | 2026-06-16 06:18:48 | 271 | 36 |

## Run Mapping

| setup | group | experiment | hash | created_at | hard_best_epoch_index | medium_best_epoch_index |
|---|---|---|---|---:|---:|---:|
| `1000_10_6e-5` | `scratch` | `scratch_1000_10_6e-5` | `e858bba7cb6a45bc9cacead0` | 2026-06-05 15:24:10 | 7 | 10 |
| `1000_10_6e-5` | `bp-const` | `tune_bp-const_1000_10_6e-5` | `89cb3a4739a449b990df1e05` | 2026-06-07 16:49:07 | 7 | 3 |
| `1000_10_6e-5` | `bp-unconst` | `tune_bp-unconst_1000_10_6e-5` | `ccaa7e35a5e8483ca3634b28` | 2026-06-07 19:44:18 | 6 | 10 |
| `1000_10_6e-5` | `c-const` | `tune_c-const_1000_10_6e-5` | `a4382648750e47dd8cb562d8` | 2026-06-17 08:45:22 | 0 | 3 |
| `1000_10_6e-5` | `c-const-norm` | `tune_c-const-norm_1000_10_6e-5` | `3ce06a38f7e34069bf30a3b1` | 2026-06-17 21:55:35 | 4 | 3 |
| `1000_10_6e-5` | `c-unconst` | `tune_c-unconst_1000_10_6e-5` | `babd19a818e54e26b941c5ab` | 2026-06-17 12:40:38 | 4 | 3 |
| `1000_10_2e-4` | `scratch` | `scratch_1000_10_2e-4` | `3d0b395faa354d438d413af7` | 2026-06-05 15:24:10 | 4 | 4 |
| `1000_10_2e-4` | `bp-const` | `tune_bp-const_1000_10_2e-4` | `46b5f33d87a74d5587173056` | 2026-06-07 16:38:44 | 6 | 3 |
| `1000_10_2e-4` | `bp-unconst` | `tune_bp-unconst_1000_10_2e-4` | `6c43157db51f4b3c94b470be` | 2026-06-07 19:23:37 | 6 | 10 |
| `1000_10_2e-4` | `c-const` | `tune_c-const_1000_10_2e-4` | `ffecd51655ae401c94104385` | 2026-06-17 08:33:58 | 4 | 3 |
| `1000_10_2e-4` | `c-const-norm` | `tune_c-const-norm_1000_10_2e-4` | `c7b83855161c411c956bb2d4` | 2026-06-17 21:44:38 | 4 | 3 |
| `1000_10_2e-4` | `c-unconst` | `tune_c-unconst_1000_10_2e-4` | `7703e8d468954d12bf009065` | 2026-06-17 12:40:37 | 11 | 3 |
| `1000_10_6e-4` | `scratch` | `scratch_1000_10_6e-4` | `e4ae1a77b76a4ccf80727789` | 2026-06-05 15:24:11 | 8 | 8 |
| `1000_10_6e-4` | `bp-const` | `tune_bp-const_1000_10_6e-4` | `d6d018aec6c848eeb1470b98` | 2026-06-07 16:38:44 | 7 | 3 |
| `1000_10_6e-4` | `bp-unconst` | `tune_bp-unconst_1000_10_6e-4` | `99df06177a6c4c9aaa37e185` | 2026-06-07 19:33:59 | 7 | 3 |
| `1000_10_6e-4` | `c-const` | `tune_c-const_1000_10_6e-4` | `d60949bf2f3d4986bdd6d43b` | 2026-06-17 08:39:00 | 6 | 3 |
| `1000_10_6e-4` | `c-const-norm` | `tune_c-const-norm_1000_10_6e-4` | `350e440c1e37456c848712a2` | 2026-06-17 21:55:04 | 4 | 3 |
| `1000_10_6e-4` | `c-unconst` | `tune_c-unconst_1000_10_6e-4` | `29c17bcc06f14943918a249a` | 2026-06-17 12:40:38 | 11 | 3 |
| `1000_20_6e-5` | `scratch` | `scratch_1000_20_6e-5` | `efd4d7a105e746cc81a4e7b7` | 2026-06-05 15:34:24 | 16 | 5 |
| `1000_20_6e-5` | `bp-const` | `tune_bp-const_1000_20_6e-5` | `5eaf15e710bf4733a8cd8395` | 2026-06-07 17:06:57 | 15 | 13 |
| `1000_20_6e-5` | `bp-unconst` | `tune_bp-unconst_1000_20_6e-5` | `98915df94e0749209e7a0549` | 2026-06-07 19:59:16 | 14 | 3 |
| `1000_20_6e-5` | `c-const` | `tune_c-const_1000_20_6e-5` | `16b8f98c308a4913951cc6cc` | 2026-06-17 09:58:26 | 0 | 6 |
| `1000_20_6e-5` | `c-const-norm` | `tune_c-const-norm_1000_20_6e-5` | `4716bfe4455a47cc9b61902a` | 2026-06-17 23:10:05 | 7 | 4 |
| `1000_20_6e-5` | `c-unconst` | `tune_c-unconst_1000_20_6e-5` | `44b13c31195443cf88bd6d0f` | 2026-06-17 12:51:11 | 7 | 6 |
| `1000_20_2e-4` | `scratch` | `scratch_1000_20_2e-4` | `8903bc5d58f2493889099dd5` | 2026-06-05 15:34:02 | 0 | 0 |
| `1000_20_2e-4` | `bp-const` | `tune_bp-const_1000_20_2e-4` | `248a7ad3119e43f29bb61a34` | 2026-06-07 16:49:07 | 4 | 3 |
| `1000_20_2e-4` | `bp-unconst` | `tune_bp-unconst_1000_20_2e-4` | `1e75370e05844962892a66f2` | 2026-06-07 19:53:30 | 4 | 10 |
| `1000_20_2e-4` | `c-const` | `tune_c-const_1000_20_2e-4` | `a16406ca800d46a4ba48717f` | 2026-06-17 09:24:58 | 7 | 10 |
| `1000_20_2e-4` | `c-const-norm` | `tune_c-const-norm_1000_20_2e-4` | `3fe3abccf83d4275982baf40` | 2026-06-17 22:52:07 | 7 | 6 |
| `1000_20_2e-4` | `c-unconst` | `tune_c-unconst_1000_20_2e-4` | `e30720de8a534e94a590caa2` | 2026-06-17 12:50:55 | 20 | 3 |
| `1000_20_6e-4` | `scratch` | `scratch_1000_20_6e-4` | `2ab1dadc76f3498885e06e77` | 2026-06-05 15:34:24 | 0 | 0 |
| `1000_20_6e-4` | `bp-const` | `tune_bp-const_1000_20_6e-4` | `d7444115200949dfb5b5481f` | 2026-06-07 16:59:30 | 14 | 10 |
| `1000_20_6e-4` | `bp-unconst` | `tune_bp-unconst_1000_20_6e-4` | `48a4edbdcde54eeca11cfa48` | 2026-06-07 19:54:39 | 13 | 3 |
| `1000_20_6e-4` | `c-const` | `tune_c-const_1000_20_6e-4` | `00de5afb81214b6ba1e3283c` | 2026-06-17 09:44:02 | 14 | 10 |
| `1000_20_6e-4` | `c-const-norm` | `tune_c-const-norm_1000_20_6e-4` | `c5a8672495744930a3bd19c8` | 2026-06-17 23:00:28 | 7 | 10 |
| `1000_20_6e-4` | `c-unconst` | `tune_c-unconst_1000_20_6e-4` | `a3ef4ee53f60410390ede9d9` | 2026-06-17 12:51:02 | 16 | 3 |
| `1000_40_6e-5` | `scratch` | `scratch_1000_40_6e-5` | `06d4800236ec47c582b7d546` | 2026-06-05 15:51:56 | 36 | 29 |
| `1000_40_6e-5` | `bp-const` | `tune_bp-const_1000_40_6e-5` | `fa93e4d06bc745c1ae37bf71` | 2026-06-07 17:49:41 | 31 | 21 |
| `1000_40_6e-5` | `bp-unconst` | `tune_bp-unconst_1000_40_6e-5` | `19e31313b2f74fd7ac391d3b` | 2026-06-07 20:17:00 | 27 | 14 |
| `1000_40_6e-5` | `c-const` | `tune_c-const_1000_40_6e-5` | `1ea663b7aafd40aeb1b49ac9` | 2026-06-17 09:58:14 | 2 | 41 |
| `1000_40_6e-5` | `c-const-norm` | `tune_c-const-norm_1000_40_6e-5` | `aa03862b354c4a4c9b196672` | 2026-06-17 23:39:05 | 8 | 21 |
| `1000_40_6e-5` | `c-unconst` | `tune_c-unconst_1000_40_6e-5` | `db2ce463fde24c4486251583` | 2026-06-17 13:09:15 | 15 | 1 |
| `1000_40_2e-4` | `scratch` | `scratch_1000_40_2e-4` | `bef140a919a1414fb9c9411f` | 2026-06-05 15:50:59 | 7 | 7 |
| `1000_40_2e-4` | `bp-const` | `tune_bp-const_1000_40_2e-4` | `480250d7341e439ebd419152` | 2026-06-07 17:17:20 | 27 | 36 |
| `1000_40_2e-4` | `bp-unconst` | `tune_bp-unconst_1000_40_2e-4` | `4d73a61664284e479034b076` | 2026-06-07 20:11:07 | 11 | 0 |
| `1000_40_2e-4` | `c-const` | `tune_c-const_1000_40_2e-4` | `dd0fef5be3224ffc9678da0c` | 2026-06-17 09:58:20 | 15 | 21 |
| `1000_40_2e-4` | `c-const-norm` | `tune_c-const-norm_1000_40_2e-4` | `6cf091fb49bd490d94444bc7` | 2026-06-17 23:19:21 | 11 | 21 |
| `1000_40_2e-4` | `c-unconst` | `tune_c-unconst_1000_40_2e-4` | `e80a93c2b68a429e8516f51a` | 2026-06-17 13:00:34 | 39 | 35 |
| `1000_40_6e-4` | `scratch` | `scratch_1000_40_6e-4` | `c3c1c572646144f6930a5b02` | 2026-06-05 15:51:49 | 10 | 1 |
| `1000_40_6e-4` | `bp-const` | `tune_bp-const_1000_40_6e-4` | `05fecc52ac9040b4867052bf` | 2026-06-07 17:24:39 | 27 | 35 |
| `1000_40_6e-4` | `bp-unconst` | `tune_bp-unconst_1000_40_6e-4` | `a703dd0bc6734ab4bdfd473a` | 2026-06-07 20:12:17 | 11 | 6 |
| `1000_40_6e-4` | `c-const` | `tune_c-const_1000_40_6e-4` | `89af51402925411abcdc0c1a` | 2026-06-17 09:58:21 | 31 | 6 |
| `1000_40_6e-4` | `c-const-norm` | `tune_c-const-norm_1000_40_6e-4` | `4e52a23b614144c7a37eec72` | 2026-06-17 23:28:08 | 15 | 14 |
| `1000_40_6e-4` | `c-unconst` | `tune_c-unconst_1000_40_6e-4` | `e3fe8b291b6549bdac03eaf9` | 2026-06-17 13:03:49 | 31 | 35 |
| `1000_80_6e-5` | `scratch` | `scratch_1000_80_6e-5` | `f870e30a8303432998f15ebb` | 2026-06-05 16:25:00 | 32 | 32 |
| `1000_80_6e-5` | `bp-const` | `tune_bp-const_1000_80_6e-5` | `26934de5b8514fc9936c9592` | 2026-06-07 18:58:14 | 14 | 81 |
| `1000_80_6e-5` | `bp-unconst` | `tune_bp-unconst_1000_80_6e-5` | `9ce4a18946ae4ea7a3bd4b97` | 2026-06-07 20:49:15 | 61 | 38 |
| `1000_80_6e-5` | `c-const` | `tune_c-const_1000_80_6e-5` | `1d704aae0aaf4fc5becf6c23` | 2026-06-17 10:11:50 | 2 | 38 |
| `1000_80_6e-5` | `c-const-norm` | `tune_c-const-norm_1000_80_6e-5` | `9ed390472bf7456a928953c6` | 2026-06-18 00:01:10 | 16 | 50 |
| `1000_80_6e-5` | `c-unconst` | `tune_c-unconst_1000_80_6e-5` | `ece300f0038c4dd9ba3a60f4` | 2026-06-17 13:33:56 | 34 | 38 |
| `1000_80_2e-4` | `scratch` | `scratch_1000_80_2e-4` | `47952fec49034f44bd3c6f2f` | 2026-06-05 16:23:06 | 39 | 51 |
| `1000_80_2e-4` | `bp-const` | `tune_bp-const_1000_80_2e-4` | `66f0bfdb15304bff8b2c466d` | 2026-06-07 17:56:42 | 12 | 81 |
| `1000_80_2e-4` | `bp-unconst` | `tune_bp-unconst_1000_80_2e-4` | `91ec22b632ee4080b33f126d` | 2026-06-07 20:43:34 | 24 | 38 |
| `1000_80_2e-4` | `c-const` | `tune_c-const_1000_80_2e-4` | `f886264795bb43a4aa990d0c` | 2026-06-17 09:58:15 | 34 | 15 |
| `1000_80_2e-4` | `c-const-norm` | `tune_c-const-norm_1000_80_2e-4` | `75db1c03d26146f4a1aa1693` | 2026-06-17 23:40:08 | 24 | 46 |
| `1000_80_2e-4` | `c-unconst` | `tune_c-unconst_1000_80_2e-4` | `ef0634e26d6b450287d0c2bb` | 2026-06-17 13:09:35 | 78 | 38 |
| `1000_80_6e-4` | `scratch` | `scratch_1000_80_6e-4` | `32e3b7fe00cc40ba894463a9` | 2026-06-05 16:24:42 | 39 | 81 |
| `1000_80_6e-4` | `bp-const` | `tune_bp-const_1000_80_6e-4` | `211242ea859b49a5b9ed3236` | 2026-06-07 18:22:05 | 48 | 17 |
| `1000_80_6e-4` | `bp-unconst` | `tune_bp-unconst_1000_80_6e-4` | `9972f04e57c74fd08af5e4d7` | 2026-06-07 20:44:40 | 24 | 81 |
| `1000_80_6e-4` | `c-const` | `tune_c-const_1000_80_6e-4` | `f9d600c2d17d4d1b914bdc33` | 2026-06-17 10:02:39 | 2 | 15 |
| `1000_80_6e-4` | `c-const-norm` | `tune_c-const-norm_1000_80_6e-4` | `cd1f733db19144e3a00d920c` | 2026-06-17 23:53:22 | 34 | 15 |
| `1000_80_6e-4` | `c-unconst` | `tune_c-unconst_1000_80_6e-4` | `e632313df56a4410ae618575` | 2026-06-17 13:09:39 | 67 | 46 |
| `1000_100_6e-5` | `scratch` | `scratch_1000_100_6e-5` | `c97e233c41924970a85075a5` | 2026-06-12 07:07:38 | 10 | 98 |
| `1000_100_6e-5` | `bp-const` | `tune_bp-const_1000_100_6e-5` | `9116c50778054e709c04423a` | 2026-06-17 10:29:52 | 83 | 98 |
| `1000_100_6e-5` | `bp-unconst` | `tune_bp-unconst_1000_100_6e-5` | `9970b5d23d67455cbc1a5c92` | 2026-06-17 10:54:04 | 77 | 98 |
| `1000_100_6e-5` | `c-const` | `tune_c-const_1000_100_6e-5` | `b472d6ed4ee8429880519378` | 2026-06-17 07:49:57 | 3 | 46 |
| `1000_100_6e-5` | `c-const-norm` | `tune_c-const-norm_1000_100_6e-5` | `6a76ef62624a49d7841d4167` | 2026-06-17 21:41:22 | 20 | 98 |
| `1000_100_6e-5` | `c-unconst` | `tune_c-unconst_1000_100_6e-5` | `8ac16db452f048c29c8a7566` | 2026-06-17 12:40:37 | 40 | 98 |
| `1000_100_2e-4` | `scratch` | `scratch_1000_100_2e-4` | `3d3082385afb4ef5a0757dae` | 2026-06-12 07:07:55 | 1 | 4 |
| `1000_100_2e-4` | `bp-const` | `tune_bp-const_1000_100_2e-4` | `643ef08469a54f38a24f439b` | 2026-06-17 10:17:02 | 30 | 98 |
| `1000_100_2e-4` | `bp-unconst` | `tune_bp-unconst_1000_100_2e-4` | `e9d382b05dae4b02a72d3cd1` | 2026-06-17 10:40:06 | 30 | 98 |
| `1000_100_2e-4` | `c-const` | `tune_c-const_1000_100_2e-4` | `a2383146b57d4cdd949083b8` | 2026-06-17 07:14:28 | 40 | 95 |
| `1000_100_2e-4` | `c-const-norm` | `tune_c-const-norm_1000_100_2e-4` | `efca1241ec5d4c57983d8c2b` | 2026-06-17 20:26:11 | 30 | 64 |
| `1000_100_2e-4` | `c-unconst` | `tune_c-unconst_1000_100_2e-4` | `82497d4def6649c596b0b385` | 2026-06-17 12:33:10 | 97 | 98 |
| `1000_100_6e-4` | `scratch` | `scratch_1000_100_6e-4` | `ccfa8d4baca8448796cf4207` | 2026-06-12 07:07:38 | 67 | 47 |
| `1000_100_6e-4` | `bp-const` | `tune_bp-const_1000_100_6e-4` | `ab4623a99a35452780497a1a` | 2026-06-17 10:26:20 | 20 | 101 |
| `1000_100_6e-4` | `bp-unconst` | `tune_bp-unconst_1000_100_6e-4` | `2df1580e009c4320a8bde1de` | 2026-06-17 10:54:04 | 33 | 59 |
| `1000_100_6e-4` | `c-const` | `tune_c-const_1000_100_6e-4` | `e1eb9f83f46e4d16af2aa2ab` | 2026-06-17 07:17:09 | 3 | 101 |
| `1000_100_6e-4` | `c-const-norm` | `tune_c-const-norm_1000_100_6e-4` | `ef3f6fa131f04aa99d012b00` | 2026-06-17 21:23:15 | 83 | 63 |
| `1000_100_6e-4` | `c-unconst` | `tune_c-unconst_1000_100_6e-4` | `468589deee7f4c44aedfb3ca` | 2026-06-17 12:33:10 | 71 | 101 |
| `1000_120_6e-5` | `scratch` | `scratch_1000_120_6e-5` | `8decc487efcd414e8d353f3d` | 2026-06-12 07:07:45 | 56 | 22 |
| `1000_120_6e-5` | `bp-const` | `tune_bp-const_1000_120_6e-5` | `16b172fb5e204676bf162ea6` | 2026-06-17 10:31:45 | 100 | 121 |
| `1000_120_6e-5` | `bp-unconst` | `tune_bp-unconst_1000_120_6e-5` | `4207c5a6d3444981876e22c7` | 2026-06-17 11:06:56 | 94 | 118 |
| `1000_120_6e-5` | `c-const` | `tune_c-const_1000_120_6e-5` | `9257346aba2f46008295a1bf` | 2026-06-17 09:07:28 | 3 | 54 |
| `1000_120_6e-5` | `c-const-norm` | `tune_c-const-norm_1000_120_6e-5` | `6790d5ee10f44549b4281056` | 2026-06-17 22:42:42 | 23 | 118 |
| `1000_120_6e-5` | `c-unconst` | `tune_c-unconst_1000_120_6e-5` | `ebcda23e46494ad09da2d7d8` | 2026-06-17 12:50:47 | 46 | 118 |
| `1000_120_2e-4` | `scratch` | `scratch_1000_120_2e-4` | `ac539cade8db41979bdbc0d7` | 2026-06-12 07:07:45 | 86 | 58 |
| `1000_120_2e-4` | `bp-const` | `tune_bp-const_1000_120_2e-4` | `9d56d8fc83ab4286afbfaff7` | 2026-06-17 10:30:41 | 86 | 68 |
| `1000_120_2e-4` | `bp-unconst` | `tune_bp-unconst_1000_120_2e-4` | `a2723557ab2c49898f25c679` | 2026-06-17 10:54:04 | 35 | 68 |
| `1000_120_2e-4` | `c-const` | `tune_c-const_1000_120_2e-4` | `d64d82a48b104cf784342255` | 2026-06-17 08:50:34 | 46 | 7 |
| `1000_120_2e-4` | `c-const-norm` | `tune_c-const-norm_1000_120_2e-4` | `8eb15104ecbd4990bd0e9c5a` | 2026-06-17 22:06:01 | 35 | 118 |
| `1000_120_2e-4` | `c-unconst` | `tune_c-unconst_1000_120_2e-4` | `0eb7b02e7b56468a97a518f7` | 2026-06-17 12:40:38 | 117 | 68 |
| `1000_120_6e-4` | `scratch` | `scratch_1000_120_6e-4` | `9899eb3af5f643478c87deb9` | 2026-06-12 07:07:45 | 62 | 103 |
| `1000_120_6e-4` | `bp-const` | `tune_bp-const_1000_120_6e-4` | `c8a8ead47b5141ac9c1f8c27` | 2026-06-17 10:31:15 | 38 | 7 |
| `1000_120_6e-4` | `bp-unconst` | `tune_bp-unconst_1000_120_6e-4` | `4b3b7099d302407397e12049` | 2026-06-17 11:01:58 | 35 | 7 |
| `1000_120_6e-4` | `c-const` | `tune_c-const_1000_120_6e-4` | `49f123232bf048189ae308da` | 2026-06-17 08:56:23 | 100 | 118 |
| `1000_120_6e-4` | `c-const-norm` | `tune_c-const-norm_1000_120_6e-4` | `6b283c00633b49d9bdddf013` | 2026-06-17 22:06:35 | 46 | 73 |
| `1000_120_6e-4` | `c-unconst` | `tune_c-unconst_1000_120_6e-4` | `08c52f092eae4d2790b65222` | 2026-06-17 12:41:25 | 84 | 68 |
