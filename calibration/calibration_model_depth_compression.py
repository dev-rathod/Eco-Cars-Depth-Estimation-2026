# Importing the required libraries for data processing and file management
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
from scipy import stats
import time

# List of training dataset segments to process
train_dataset = ['segment-1906113358876584689_1359_560_1379_560_with_camera_labels', 'segment-191862526745161106_1400_000_1420_000_with_camera_labels', 'segment-1936395688683397781_2580_000_2600_000_with_camera_labels', 'segment-1943605865180232897_680_000_700_000_with_camera_labels', 'segment-2094681306939952000_2972_300_2992_300_with_camera_labels', 'segment-2105808889850693535_2295_720_2315_720_with_camera_labels', 'segment-2218963221891181906_4360_000_4380_000_with_camera_labels', 'segment-2257381802419655779_820_000_840_000_with_camera_labels', 'segment-2308204418431899833_3575_000_3595_000_with_camera_labels', 'segment-2335854536382166371_2709_426_2729_426_with_camera_labels', 'segment-2363225200168330815_760_000_780_000_with_camera_labels', 'segment-2367305900055174138_1881_827_1901_827_with_camera_labels', 'segment-2374138435300423201_2600_000_2620_000_with_camera_labels', 'segment-2383902674438058857_4420_000_4440_000_with_camera_labels', 'segment-2506799708748258165_6455_000_6475_000_with_camera_labels', 'segment-2551868399007287341_3100_000_3120_000_with_camera_labels', 'segment-2601205676330128831_4880_000_4900_000_with_camera_labels', 'segment-260994483494315994_2797_545_2817_545_with_camera_labels', 'segment-2624187140172428292_73_000_93_000_with_camera_labels', 'segment-2709541197299883157_1140_000_1160_000_with_camera_labels', 'segment-271338158136329280_2541_070_2561_070_with_camera_labels', 'segment-2714318267497393311_480_000_500_000_with_camera_labels', 'segment-272435602399417322_2884_130_2904_130_with_camera_labels', 'segment-2736377008667623133_2676_410_2696_410_with_camera_labels', 'segment-2795127582672852315_4140_000_4160_000_with_camera_labels', 'segment-2830680430134047327_1720_000_1740_000_with_camera_labels', 'segment-2834723872140855871_1615_000_1635_000_with_camera_labels', 'segment-2906594041697319079_3040_000_3060_000_with_camera_labels', 'segment-2942662230423855469_880_000_900_000_with_camera_labels', 'segment-3015436519694987712_1300_000_1320_000_with_camera_labels', 'segment-3039251927598134881_1240_610_1260_610_with_camera_labels', 'segment-3077229433993844199_1080_000_1100_000_with_camera_labels', 'segment-30779396576054160_1880_000_1900_000_with_camera_labels', 'segment-3122599254941105215_2980_000_3000_000_with_camera_labels', 'segment-3126522626440597519_806_440_826_440_with_camera_labels', 'segment-3275806206237593341_1260_000_1280_000_with_camera_labels', 'segment-3328513486129168664_2080_000_2100_000_with_camera_labels', 'segment-3341890853207909601_1020_000_1040_000_with_camera_labels', 'segment-3400465735719851775_1400_000_1420_000_with_camera_labels', 'segment-3459095437766396887_1600_000_1620_000_with_camera_labels', 'segment-346889320598157350_798_187_818_187_with_camera_labels', 'segment-3485136235103477552_600_000_620_000_with_camera_labels', 'segment-3510690431623954420_7700_000_7720_000_with_camera_labels', 'segment-3522804493060229409_3400_000_3420_000_with_camera_labels', 'segment-3577352947946244999_3980_000_4000_000_with_camera_labels', 'segment-3645211352574995740_3540_000_3560_000_with_camera_labels', 'segment-3651243243762122041_3920_000_3940_000_with_camera_labels', 'segment-365416647046203224_1080_000_1100_000_with_camera_labels', 'segment-366934253670232570_2229_530_2249_530_with_camera_labels', 'segment-3731719923709458059_1540_000_1560_000_with_camera_labels', 'segment-3915587593663172342_10_000_30_000_with_camera_labels', 'segment-39847154216997509_6440_000_6460_000_with_camera_labels', 'segment-4008112367880337022_3680_000_3700_000_with_camera_labels', 'segment-4013125682946523088_3540_000_3560_000_with_camera_labels', 'segment-4037952268810331899_2420_000_2440_000_with_camera_labels', 'segment-4045613324047897473_940_000_960_000_with_camera_labels', 'segment-4054036670499089296_2300_000_2320_000_with_camera_labels', 'segment-4140965781175793864_460_000_480_000_with_camera_labels', 'segment-4195774665746097799_7300_960_7320_960_with_camera_labels', 'segment-4246537812751004276_1560_000_1580_000_with_camera_labels', 'segment-4409585400955983988_3500_470_3520_470_with_camera_labels', 'segment-4423389401016162461_4235_900_4255_900_with_camera_labels', 'segment-4426410228514970291_1620_000_1640_000_with_camera_labels', 'segment-447576862407975570_4360_000_4380_000_with_camera_labels', 'segment-4490196167747784364_616_569_636_569_with_camera_labels', 'segment-4575389405178805994_4900_000_4920_000_with_camera_labels', 'segment-4593468568253300598_1620_000_1640_000_with_camera_labels', 'segment-4612525129938501780_340_000_360_000_with_camera_labels', 'segment-4632556232973423919_2940_000_2960_000_with_camera_labels', 'segment-4690718861228194910_1980_000_2000_000_with_camera_labels', 'segment-4759225533437988401_800_000_820_000_with_camera_labels', 'segment-4764167778917495793_860_000_880_000_with_camera_labels', 'segment-4816728784073043251_5273_410_5293_410_with_camera_labels', 'segment-4854173791890687260_2880_000_2900_000_with_camera_labels', 'segment-4916600861562283346_3880_000_3900_000_with_camera_labels', 'segment-5026942594071056992_3120_000_3140_000_with_camera_labels', 'segment-5046614299208670619_1760_000_1780_000_with_camera_labels', 'segment-5154724129640787887_4840_000_4860_000_with_camera_labels', 'segment-5183174891274719570_3464_030_3484_030_with_camera_labels', 'segment-5289247502039512990_2640_000_2660_000_with_camera_labels', 'segment-5302885587058866068_320_000_340_000_with_camera_labels', 'segment-5372281728627437618_2005_000_2025_000_with_camera_labels', 'segment-5373876050695013404_3817_170_3837_170_with_camera_labels', 'segment-5444585006397501511_160_000_180_000_with_camera_labels', 'segment-5574146396199253121_6759_360_6779_360_with_camera_labels', 'segment-5585555620508986875_720_000_740_000_with_camera_labels', 'segment-5638240639308158118_4220_000_4240_000_with_camera_labels', 'segment-5648007586817904385_3220_000_3240_000_with_camera_labels', 'segment-5683383258122801095_1040_000_1060_000_with_camera_labels', 'segment-5764319372514665214_2480_000_2500_000_with_camera_labels', 'segment-5772016415301528777_1400_000_1420_000_with_camera_labels', 'segment-5810494922060252082_3720_000_3740_000_with_camera_labels', 'segment-5832416115092350434_60_000_80_000_with_camera_labels', 'segment-5847910688643719375_180_000_200_000_with_camera_labels', 'segment-5927928428387529213_1640_000_1660_000_with_camera_labels', 'segment-5990032395956045002_6600_000_6620_000_with_camera_labels', 'segment-5993415832220804439_1020_000_1040_000_with_camera_labels', 'segment-6001094526418694294_4609_470_4629_470_with_camera_labels', 'segment-6074871217133456543_1000_000_1020_000_with_camera_labels', 'segment-6079272500228273268_2480_000_2500_000_with_camera_labels', 'segment-614453665074997770_1060_000_1080_000_with_camera_labels', 'segment-6161542573106757148_585_030_605_030_with_camera_labels', 'segment-6174376739759381004_3240_000_3260_000_with_camera_labels', 'segment-6183008573786657189_5414_000_5434_000_with_camera_labels', 'segment-6228701001600487900_720_000_740_000_with_camera_labels', 'segment-6259508587655502768_780_000_800_000_with_camera_labels', 'segment-6278307160249415497_1700_000_1720_000_with_camera_labels', 'segment-6324079979569135086_2372_300_2392_300_with_camera_labels', 'segment-6491418762940479413_6520_000_6540_000_with_camera_labels', 'segment-6503078254504013503_3440_000_3460_000_with_camera_labels', 'segment-662188686397364823_3248_800_3268_800_with_camera_labels', 'segment-6637600600814023975_2235_000_2255_000_with_camera_labels', 'segment-6680764940003341232_2260_000_2280_000_with_camera_labels', 'segment-6707256092020422936_2352_392_2372_392_with_camera_labels', 'segment-684234579698396203_2540_000_2560_000_with_camera_labels', 'segment-6862795755554967162_2280_000_2300_000_with_camera_labels', 'segment-6922883602463663456_2220_000_2240_000_with_camera_labels', 'segment-7119831293178745002_1094_720_1114_720_with_camera_labels', 'segment-7163140554846378423_2717_820_2737_820_with_camera_labels', 'segment-7240042450405902042_580_000_600_000_with_camera_labels', 'segment-7247823803417339098_2320_000_2340_000_with_camera_labels', 'segment-7253952751374634065_1100_000_1120_000_with_camera_labels', 'segment-7435516779413778621_4440_000_4460_000_with_camera_labels', 'segment-7493781117404461396_2140_000_2160_000_with_camera_labels', 'segment-7511993111693456743_3880_000_3900_000_with_camera_labels', 'segment-7650923902987369309_2380_000_2400_000_with_camera_labels', 'segment-7732779227944176527_2120_000_2140_000_with_camera_labels', 'segment-7799643635310185714_680_000_700_000_with_camera_labels', 'segment-7844300897851889216_500_000_520_000_with_camera_labels', 'segment-7855150647548977812_3900_000_3920_000_with_camera_labels', 'segment-7886090431228432618_1060_000_1080_000_with_camera_labels', 'segment-792520390268391604_780_000_800_000_with_camera_labels', 'segment-7932945205197754811_780_000_800_000_with_camera_labels', 'segment-7988627150403732100_1487_540_1507_540_with_camera_labels', 'segment-8079607115087394458_1240_000_1260_000_with_camera_labels', 'segment-8085856200343017603_4120_000_4140_000_with_camera_labels', 'segment-8133434654699693993_1162_020_1182_020_with_camera_labels', 'segment-8137195482049459160_3100_000_3120_000_with_camera_labels', 'segment-8197312656120253218_3120_000_3140_000_with_camera_labels', 'segment-8229317157758012712_3860_000_3880_000_with_camera_labels', 'segment-8249122135171526629_520_000_540_000_with_camera_labels', 'segment-8302000153252334863_6020_000_6040_000_with_camera_labels', 'segment-8331804655557290264_4351_740_4371_740_with_camera_labels', 'segment-8398516118967750070_3958_000_3978_000_with_camera_labels', 'segment-8506432817378693815_4860_000_4880_000_with_camera_labels', 'segment-8566480970798227989_500_000_520_000_with_camera_labels', 'segment-8623236016759087157_3500_000_3520_000_with_camera_labels', 'segment-8679184381783013073_7740_000_7760_000_with_camera_labels', 'segment-8684065200957554260_2700_000_2720_000_with_camera_labels', 'segment-8688567562597583972_940_000_960_000_with_camera_labels', 'segment-8845277173853189216_3828_530_3848_530_with_camera_labels', 'segment-8888517708810165484_1549_770_1569_770_with_camera_labels', 'segment-8907419590259234067_1960_000_1980_000_with_camera_labels', 'segment-8920841445900141920_1700_000_1720_000_with_camera_labels', 'segment-89454214745557131_3160_000_3180_000_with_camera_labels', 'segment-8956556778987472864_3404_790_3424_790_with_camera_labels', 'segment-8993680275027614595_2520_000_2540_000_with_camera_labels', 'segment-902001779062034993_2880_000_2900_000_with_camera_labels', 'segment-9024872035982010942_2578_810_2598_810_with_camera_labels', 'segment-9041488218266405018_6454_030_6474_030_with_camera_labels', 'segment-9114112687541091312_1100_000_1120_000_with_camera_labels', 'segment-9145030426583202228_1060_000_1080_000_with_camera_labels', 'segment-9164052963393400298_4692_970_4712_970_with_camera_labels', 'segment-9231652062943496183_1740_000_1760_000_with_camera_labels', 'segment-9243656068381062947_1297_428_1317_428_with_camera_labels', 'segment-9265793588137545201_2981_960_3001_960_with_camera_labels', 'segment-933621182106051783_4160_000_4180_000_with_camera_labels', 'segment-9350911198443552989_680_000_700_000_with_camera_labels', 'segment-9355489589631690177_4800_000_4820_000_with_camera_labels', 'segment-9443948810903981522_6538_870_6558_870_with_camera_labels', 'segment-9472420603764812147_850_000_870_000_with_camera_labels', 'segment-9579041874842301407_1300_000_1320_000_with_camera_labels', 'segment-9584760613582366524_1620_000_1640_000_with_camera_labels', 'segment-967082162553397800_5102_900_5122_900_with_camera_labels', 'segment-10084636266401282188_1120_000_1140_000_with_camera_labels', 'segment-10149575340910243572_2720_000_2740_000_with_camera_labels', 'segment-10161761842905385678_760_000_780_000_with_camera_labels', 'segment-10203656353524179475_7625_000_7645_000_with_camera_labels', 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels', 'segment-10247954040621004675_2180_000_2200_000_with_camera_labels', 'segment-10289507859301986274_4200_000_4220_000_with_camera_labels', 'segment-10335539493577748957_1372_870_1392_870_with_camera_labels', 'segment-10359308928573410754_720_000_740_000_with_camera_labels', 'segment-10410418118434245359_5140_000_5160_000_with_camera_labels', 'segment-10448102132863604198_472_000_492_000_with_camera_labels', 'segment-10488772413132920574_680_000_700_000_with_camera_labels', 'segment-10504764403039842352_460_000_480_000_with_camera_labels', 'segment-10534368980139017457_4480_000_4500_000_with_camera_labels', 'segment-10689101165701914459_2072_300_2092_300_with_camera_labels', 'segment-1071392229495085036_1844_790_1864_790_with_camera_labels', 'segment-10802932587105534078_1280_000_1300_000_with_camera_labels', 'segment-10837554759555844344_6525_000_6545_000_with_camera_labels', 'segment-10868756386479184868_3000_000_3020_000_with_camera_labels', 'segment-10940141908690367388_4420_000_4440_000_with_camera_labels', 'segment-10980133015080705026_780_000_800_000_with_camera_labels', 'segment-10998289306141768318_1280_000_1300_000_with_camera_labels', 'segment-11037651371539287009_77_670_97_670_with_camera_labels', 'segment-11048712972908676520_545_000_565_000_with_camera_labels', 'segment-1105338229944737854_1280_000_1300_000_with_camera_labels', 'segment-11096867396355523348_1460_000_1480_000_with_camera_labels', 'segment-11356601648124485814_409_000_429_000_with_camera_labels', 'segment-11387395026864348975_3820_000_3840_000_with_camera_labels', 'segment-11406166561185637285_1753_750_1773_750_with_camera_labels', 'segment-11434627589960744626_4829_660_4849_660_with_camera_labels', 'segment-11436803605426256250_1720_000_1740_000_with_camera_labels', 'segment-11450298750351730790_1431_750_1451_750_with_camera_labels', 'segment-11616035176233595745_3548_820_3568_820_with_camera_labels', 'segment-11660186733224028707_420_000_440_000_with_camera_labels', 'segment-11672844176539348333_4440_000_4460_000_with_camera_labels', 'segment-11867874114645674271_600_000_620_000_with_camera_labels', 'segment-11901761444769610243_556_000_576_000_with_camera_labels', 'segment-11933765568165455008_2940_000_2960_000_with_camera_labels', 'segment-11987368976578218644_1340_000_1360_000_with_camera_labels', 'segment-12056192874455954437_140_000_160_000_with_camera_labels', 'segment-12102100359426069856_3931_470_3951_470_with_camera_labels', 'segment-12134738431513647889_3118_000_3138_000_with_camera_labels', 'segment-12153647356523920032_2560_000_2580_000_with_camera_labels', 'segment-12306251798468767010_560_000_580_000_with_camera_labels', 'segment-12358364923781697038_2232_990_2252_990_with_camera_labels', 'segment-12374656037744638388_1412_711_1432_711_with_camera_labels', 'segment-12496433400137459534_120_000_140_000_with_camera_labels', 'segment-12537711031998520792_3080_000_3100_000_with_camera_labels', 'segment-12555145882162126399_1180_000_1200_000_with_camera_labels', 'segment-12657584952502228282_3940_000_3960_000_with_camera_labels', 'segment-12820461091157089924_5202_916_5222_916_with_camera_labels', 'segment-12831741023324393102_2673_230_2693_230_with_camera_labels', 'segment-12866817684252793621_480_000_500_000_with_camera_labels', 'segment-12892154548237137398_2820_000_2840_000_with_camera_labels', 'segment-12940710315541930162_2660_000_2680_000_with_camera_labels', 'segment-13034900465317073842_1700_000_1720_000_with_camera_labels', 'segment-13178092897340078601_5118_604_5138_604_with_camera_labels', 'segment-13184115878756336167_1354_000_1374_000_with_camera_labels', 'segment-13299463771883949918_4240_000_4260_000_with_camera_labels', 'segment-1331771191699435763_440_000_460_000_with_camera_labels', 'segment-13336883034283882790_7100_000_7120_000_with_camera_labels', 'segment-13347759874869607317_1540_000_1560_000_with_camera_labels', 'segment-13356997604177841771_3360_000_3380_000_with_camera_labels', 'segment-13415985003725220451_6163_000_6183_000_with_camera_labels', 'segment-13469905891836363794_4429_660_4449_660_with_camera_labels', 'segment-13573359675885893802_1985_970_2005_970_with_camera_labels', 'segment-13585389505831587326_2560_000_2580_000_with_camera_labels', 'segment-13694146168933185611_800_000_820_000_with_camera_labels', 'segment-13732041959462600641_720_000_740_000_with_camera_labels', 'segment-13748565785898537200_680_000_700_000_with_camera_labels', 'segment-1376304843325714018_3420_000_3440_000_with_camera_labels', 'segment-13781857304705519152_2740_000_2760_000_with_camera_labels', 'segment-13787943721654585343_1220_000_1240_000_with_camera_labels', 'segment-13790309965076620852_6520_000_6540_000_with_camera_labels', 'segment-13849332693800388551_960_000_980_000_with_camera_labels', 'segment-13887882285811432765_740_000_760_000_with_camera_labels', 'segment-13941626351027979229_3363_930_3383_930_with_camera_labels', 'segment-13944616099709049906_1020_000_1040_000_with_camera_labels', 'segment-13982731384839979987_1680_000_1700_000_with_camera_labels', 'segment-1405149198253600237_160_000_180_000_with_camera_labels', 'segment-14081240615915270380_4399_000_4419_000_with_camera_labels', 'segment-14107757919671295130_3546_370_3566_370_with_camera_labels', 'segment-14127943473592757944_2068_000_2088_000_with_camera_labels', 'segment-14165166478774180053_1786_000_1806_000_with_camera_labels', 'segment-1417898473608326362_2560_000_2580_000_with_camera_labels', 'segment-14188689528137485670_2660_000_2680_000_with_camera_labels', 'segment-14244512075981557183_1226_840_1246_840_with_camera_labels', 'segment-14262448332225315249_1280_000_1300_000_with_camera_labels', 'segment-14300007604205869133_1160_000_1180_000_with_camera_labels', 'segment-14333744981238305769_5658_260_5678_260_with_camera_labels', 'segment-14383152291533557785_240_000_260_000_with_camera_labels', 'segment-14386836877680112549_4460_000_4480_000_with_camera_labels', 'segment-14470988792985854683_760_000_780_000_with_camera_labels', 'segment-14486517341017504003_3406_349_3426_349_with_camera_labels', 'segment-1457696187335927618_595_027_615_027_with_camera_labels', 'segment-14586026017427828517_700_000_720_000_with_camera_labels', 'segment-14624061243736004421_1840_000_1860_000_with_camera_labels', 'segment-14631629219048194483_2720_000_2740_000_with_camera_labels', 'segment-14643284977980826278_520_000_540_000_with_camera_labels', 'segment-1464917900451858484_1960_000_1980_000_with_camera_labels', 'segment-14663356589561275673_935_195_955_195_with_camera_labels', 'segment-14687328292438466674_892_000_912_000_with_camera_labels', 'segment-14737335824319407706_1980_000_2000_000_with_camera_labels', 'segment-14739149465358076158_4740_000_4760_000_with_camera_labels', 'segment-14811410906788672189_373_113_393_113_with_camera_labels', 'segment-14918167237855418464_1420_000_1440_000_with_camera_labels', 'segment-14931160836268555821_5778_870_5798_870_with_camera_labels', 'segment-14956919859981065721_1759_980_1779_980_with_camera_labels', 'segment-15021599536622641101_556_150_576_150_with_camera_labels', 'segment-15028688279822984888_1560_000_1580_000_with_camera_labels', 'segment-1505698981571943321_1186_773_1206_773_with_camera_labels', 'segment-15096340672898807711_3765_000_3785_000_with_camera_labels', 'segment-15224741240438106736_960_000_980_000_with_camera_labels', 'segment-15272375112495403395_620_000_640_000_with_camera_labels', 'segment-15370024704033662533_1240_000_1260_000_with_camera_labels', 'segment-15396462829361334065_4265_000_4285_000_with_camera_labels', 'segment-15410814825574326536_2620_000_2640_000_with_camera_labels', 'segment-15488266120477489949_3162_920_3182_920_with_camera_labels', 'segment-15496233046893489569_4551_550_4571_550_with_camera_labels', 'segment-15611747084548773814_3740_000_3760_000_with_camera_labels', 'segment-15724298772299989727_5386_410_5406_410_with_camera_labels', 'segment-15739335479094705947_1420_000_1440_000_with_camera_labels', 'segment-15865907199900332614_760_000_780_000_with_camera_labels', 'segment-15948509588157321530_7187_290_7207_290_with_camera_labels', 'segment-15959580576639476066_5087_580_5107_580_with_camera_labels', 'segment-16050146835908439029_4500_000_4520_000_with_camera_labels', 'segment-16062780403777359835_2580_000_2600_000_with_camera_labels', 'segment-16204463896543764114_5340_000_5360_000_with_camera_labels', 'segment-16213317953898915772_1597_170_1617_170_with_camera_labels', 'segment-16229547658178627464_380_000_400_000_with_camera_labels', 'segment-16367045247642649300_3060_000_3080_000_with_camera_labels', 'segment-16418654553014119039_4340_000_4360_000_with_camera_labels', 'segment-1664548685643064400_2240_000_2260_000_with_camera_labels', 'segment-16721473705085324478_2580_000_2600_000_with_camera_labels', 'segment-16743182245734335352_1260_000_1280_000_with_camera_labels', 'segment-16751706457322889693_4475_240_4495_240_with_camera_labels', 'segment-16767575238225610271_5185_000_5205_000_with_camera_labels', 'segment-16942495693882305487_4340_000_4360_000_with_camera_labels', 'segment-16951245307634830999_1400_000_1420_000_with_camera_labels', 'segment-16979882728032305374_2719_000_2739_000_with_camera_labels', 'segment-1703056599550681101_4380_000_4400_000_with_camera_labels', 'segment-17052666463197337241_4560_000_4580_000_with_camera_labels', 'segment-17065833287841703_2980_000_3000_000_with_camera_labels', 'segment-17135518413411879545_1480_000_1500_000_with_camera_labels', 'segment-17136314889476348164_979_560_999_560_with_camera_labels', 'segment-17136775999940024630_4860_000_4880_000_with_camera_labels', 'segment-17152649515605309595_3440_000_3460_000_with_camera_labels', 'segment-17174012103392027911_3500_000_3520_000_with_camera_labels', 'segment-17212025549630306883_2500_000_2520_000_with_camera_labels', 'segment-17244566492658384963_2540_000_2560_000_with_camera_labels', 'segment-17262030607996041518_540_000_560_000_with_camera_labels', 'segment-17344036177686610008_7852_160_7872_160_with_camera_labels', 'segment-1735154401471216485_440_000_460_000_with_camera_labels', 'segment-17387485694427326992_760_000_780_000_with_camera_labels', 'segment-17539775446039009812_440_000_460_000_with_camera_labels', 'segment-17595457728136868510_860_000_880_000_with_camera_labels', 'segment-17612470202990834368_2800_000_2820_000_with_camera_labels', 'segment-17626999143001784258_2760_000_2780_000_with_camera_labels', 'segment-1765211916310163252_4400_000_4420_000_with_camera_labels', 'segment-17694030326265859208_2340_000_2360_000_with_camera_labels', 'segment-17703234244970638241_220_000_240_000_with_camera_labels', 'segment-17756183617755834457_1940_000_1960_000_with_camera_labels', 'segment-17763730878219536361_3144_635_3164_635_with_camera_labels', 'segment-17791493328130181905_1480_000_1500_000_with_camera_labels', 'segment-17792628511034220885_2360_000_2380_000_with_camera_labels', 'segment-17860546506509760757_6040_000_6060_000_with_camera_labels', 'segment-17962792089966876718_2210_933_2230_933_with_camera_labels', 'segment-18024188333634186656_1566_600_1586_600_with_camera_labels', 'segment-18045724074935084846_6615_900_6635_900_with_camera_labels', 'segment-18149616047892103767_2460_000_2480_000_with_camera_labels', 'segment-18252111882875503115_378_471_398_471_with_camera_labels', 'segment-18305329035161925340_4466_730_4486_730_with_camera_labels', 'segment-18331704533904883545_1560_000_1580_000_with_camera_labels', 'segment-18333922070582247333_320_280_340_280_with_camera_labels', 'segment-18446264979321894359_3700_000_3720_000_with_camera_labels']

# Path to the groundtruth data folder
gt_folder = Path("/home/jiale/Documents/eco_cars/data/output_all_depth")
# Path to the relative data folder
rl_folder = Path("/home/jiale/Documents/eco_cars/data/depth_anything/all_depths")

# Path to the csv file containing the extracted a, b and c values for the calibration utility
csv_file_path = Path("/home/jiale/Documents/eco_cars/data/analysis/scale_alignment_results_cleaned.csv")
calibration_details = pd.read_csv(csv_file_path)

# Temporary folder to extract the groundtruth depth files
temp_gt_extract = Path("/home/jiale/Documents/eco_cars/data/calibration/gt")
# Temporary folder to extract the relative depth files
temp_rl_extract = Path("/home/jiale/Documents/eco_cars/data/calibration/rl")

# Storing the output in a seperate location in numpy arrays
output_folder = Path("/home/jiale/Documents/eco_cars/data/calibration/output")
output_folder.mkdir(parents=True, exist_ok=True)
def noiseHandler(pred40, gt40):
    groundTruth = gt40.flatten()
    predictions = pred40.flatten()

    totalBins = 20
    numberOfBins = np.linspace(0, 1, totalBins + 1)
    numberOfBins_index = np.digitize(predictions, numberOfBins) - 1
    binsInlier = np.zeros(len(predictions), dtype=bool)
    minimum_entries = 25

    for individualBins in range(totalBins):
        binsTotalEntries = (numberOfBins_index == individualBins)
        if binsTotalEntries.sum() < minimum_entries:
            continue
        goundtruth_entry = groundTruth[binsTotalEntries]
        lowerQuartile, upperQuartile = np.percentile(goundtruth_entry, [25, 75])
        iqr = upperQuartile - lowerQuartile
        binsInlier[binsTotalEntries] = (
            (goundtruth_entry >= lowerQuartile - 1.5 * iqr) &
            (goundtruth_entry <= upperQuartile + 1.5 * iqr)
        )

    pred_inlier = predictions[binsInlier]
    gt_inlier   = groundTruth[binsInlier]

    polyGraph  = np.polyfit(pred_inlier, gt_inlier, deg=4)
    residuals  = gt_inlier - np.polyval(polyGraph, pred_inlier)
    devMask    = np.abs(stats.zscore(residuals)) < 4

    # Single combined mask into original flat space
    final_mask             = np.zeros(len(predictions), dtype=bool)
    final_mask[binsInlier] = devMask

    return final_mask   # just return the mask, caller handles arrays


def deduplicate_by_grid(u, v, x, gt, grid_size=0.007,
                         img_h=1280, img_w=1920):
    """
    One pixel per spatial grid cell. No duplicates across frames.
    Stores raw pixel coords — net scales internally.
    """
    u_norm = u / img_w
    v_norm = v / img_h

    n_u    = int(1.0 / grid_size)
    n_v    = int(1.0 / grid_size)
    cell_u = np.clip((u_norm / grid_size).astype(int), 0, n_u - 1)
    cell_v = np.clip((v_norm / grid_size).astype(int), 0, n_v - 1)
    cell_id = cell_v * n_u + cell_u

    seen  = {}
    for i in range(len(u)):
        cid = int(cell_id[i])
        if cid not in seen:
            seen[cid] = i

    idx = np.array(list(seen.values()), dtype=np.int64)
    return u[idx], v[idx], x[idx], gt[idx]

# Storing the total number of items used in a variable
total_segments = len(train_dataset)
processed_frame = 0

for segments in train_dataset:
    start_time = time.time()

    zip_file_name = segments + ".zip"
    shutil.unpack_archive(gt_folder / zip_file_name, temp_gt_extract)
    shutil.unpack_archive(rl_folder / zip_file_name, temp_rl_extract)

    gt_file_path = temp_gt_extract / "all_depths.npy"
    rl_file_path = temp_rl_extract / (segments + "_depths.npy")

    gt   = np.load(gt_file_path)
    pred = np.load(rl_file_path)

    if gt.ndim == 2:
        gt   = gt[np.newaxis]
        pred = pred[np.newaxis]

    num_frames, height, width = gt.shape

    # Raw pixel coordinate grids — no normalisation
    v_coords, u_coords = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )
    u_grid = np.broadcast_to(u_coords, gt.shape)  # (F, H, W)
    v_grid = np.broadcast_to(v_coords, gt.shape)

    # Valid mask
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (gt < 40)

    # Flatten through mask
    pred40 = pred[mask]
    gt40   = gt[mask]
    u40    = u_grid[mask]
    v40    = v_grid[mask]

    # Clean — single mask returned
    clean_mask = noiseHandler(pred40, gt40)

    u_clean    = u40[clean_mask].astype(np.float32)
    v_clean    = v40[clean_mask].astype(np.float32)
    x_clean    = pred40[clean_mask].astype(np.float32)
    gt_clean   = gt40[clean_mask].astype(np.float32)

    # Grid deduplication — removes overlapping spatial positions
    u_final, v_final, x_final, gt_final = deduplicate_by_grid(
        u_clean, v_clean, x_clean, gt_clean
    )

    # Get A, B, C for this segment
    row = calibration_details[
        calibration_details["file_id"] == segments
    ].iloc[0]
    a, b, c = float(row["a"]), float(row["b"]), float(row["c"])

    # Final fabric block — cols: u | v | x | gt | a | b | c
    n = len(u_final)
    combined = np.stack([
        u_final,
        v_final,
        x_final,
        gt_final,
        np.full(n, a, dtype=np.float32),
        np.full(n, b, dtype=np.float32),
        np.full(n, c, dtype=np.float32),
    ], axis=1)  # (N, 7) float32

    print(f"{segments}: {gt40.shape[0]:,} valid "
          f"→ {clean_mask.sum():,} clean "
          f"→ {n:,} unique  "
          f"({combined.nbytes / 1e6:.1f} MB)")

    # Save
    segment_folder = output_folder / segments
    segment_folder.mkdir(parents=True, exist_ok=True)
    np.save(segment_folder / f"{segments}.npy", combined)

    shutil.make_archive(str(output_folder / segments),
                        "zip", str(output_folder), segments)
    shutil.rmtree(segment_folder, ignore_errors=True)
    shutil.rmtree(temp_gt_extract, ignore_errors=True)
    shutil.rmtree(temp_rl_extract, ignore_errors=True)

    processed_frame+= 1
    print(f"Processed: {processed_frame}/{total_segments}")
    print(f"Estimated Time = {(total_segments-processed_frame)*(time.time()-start_time)}")