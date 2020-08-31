import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'AutoDL_ingestion_program'))
sys.path.append(os.path.join(os.getcwd(), 'AutoDL_scoring_program'))

import matplotlib.pyplot as plt
import numpy as np

config_results = {}
config_results['kakaobrain'] = {}
config_results['kakaobrain']['Chucky']  = [0.8082] * 9
config_results['kakaobrain']['Decal']   = [0.8647] * 9
config_results['kakaobrain']['Hammer']  = [0.8147] * 9
config_results['kakaobrain']['Munster'] = [0.9421] * 9
config_results['kakaobrain']['Pedro']   = [0.7948] * 9
config_results['kakaobrain']['Kraut']   = [0.6678] * 9
config_results['kakaobrain']['Katze']   = [0.8613] * 9
config_results['kakaobrain']['Kreatur'] = [0.8677] * 9

config_results['precalc'] = {}
config_results['precalc']['Chucky']  = [0.8145186273191505, 0.8574719823338359, 0.8574719823338359, 0.8574719823338359, 0.8574719823338359, 0.8574719823338359, 0.8574719823338359, 0.8574719823338359, 0.8574719823338359]
config_results['precalc']['Decal']   = [0.8251647362601252, 0.8251647362601252, 0.8251647362601252, 0.8251647362601252, 0.8251647362601252, 0.8251647362601252, 0.8251647362601252, 0.8251647362601252, 0.8251647362601252]
config_results['precalc']['Hammer']  = [0.7034904383483079, 0.7775987067175913, 0.7034904383483079, 0.7034904383483079, 0.7775987067175913, 0.7653859825215631, 0.7653859825215631, 0.7775987067175913, 0.7653859825215631]
config_results['precalc']['Munster'] = [0.9297515169008259, 0.9297515169008259, 0.9297515169008259, 0.9297515169008259, 0.9297515169008259, 0.9297515169008259, 0.9297515169008259, 0.9297515169008259, 0.9297515169008259]
config_results['precalc']['Pedro']   = [0.4111062509017694, 0.5669996108396216, 0.5669996108396216, 0.40802164751707504, 0.43253394988158267, 0.4465092848714425, 0.43253394988158267, 0.4465092848714425, 0.43253394988158267]
config_results['precalc']['Kraut']   = [0, 0, 0, 0, 0, 0, 0, 0, 0]
config_results['precalc']['Katze']   = [0.8196531123875234, 0.8109824693761103, 0.8196531123875234, 0.8109824693761103, 0.8109824693761103, 0.8109824693761103, 0.8109824693761103, 0.8109824693761103, 0.8572496925197957]
config_results['precalc']['Kreatur'] = [0.8042946005386382, 0.8042946005386382, 0.8554919708387797, 0.8554919708387797, 0.8554919708387797, 0.8554919708387797, 0.8554919708387797, 0.8554919708387797, 0.8554919708387797]

config_results['precalc_64'] = {}
config_results['precalc_64']['Chucky']  = [0, 0.8145186273191505, 0.8574719823338359, 0.8145186273191505, 0.8574719823338359, 0.8574719823338359, 0.8459374718079546, 0.8145186273191505, 0.8154023571223581]
config_results['precalc_64']['Decal']   = [0.7993100667595154, 0.8251647362601252, 0.8251647362601252, 0.7993100667595154, 0.8251647362601252, 0.8251647362601252, 0.6717918515047165, 0.8251647362601252, 0.0014424739801088523]
config_results['precalc_64']['Hammer']  = [0.6439048863484191, 0.7034904383483079, 0.7034904383483079, 0.7691526495514234, 0.7775987067175913, 0.7034904383483079, 0.7723517999669374, 0.6460024225475129, 0.636763231652871]
config_results['precalc_64']['Munster'] = [0.9297515169008259, 0.9297515169008259, 0.9297515169008259, 0.9391573289398356, 0.9297515169008259, 0.9297515169008259, 0.9288284395655821, 0.9297515169008259, 0.9427060087009058]
config_results['precalc_64']['Pedro']   = [0.5669996108396216, 0.40802164751707504, 0.4465092848714425, 0.5669996108396216, 0.43253394988158267, 0.43253394988158267, 0.5074673598534009, 0.4465092848714425, 0.01902193759692205]
config_results['precalc_64']['Kraut']   = [0, 0, 0, 0, 0, 0, 0, 0, 0]
config_results['precalc_64']['Katze']   = [0.8218017202197803, 0.8109824693761103, 0.8196531123875234, 0.8503413783820133, 0.8109824693761103, 0.85414678826315, 0.8393373625594724, 0.8235324211892344, 0.6540542193703885]
config_results['precalc_64']['Kreatur'] = [0.8454138426372891, 0.8162535078954828, 0.8066358977104926, 0.8280117451979119, 0.8554919708387797, 0.8554919708387797, 0.8496014745001329, 0.8454138426372891, 0.8496014745001329]

config_results['all_configs'] = {}
config_results['all_configs']['Chucky']  = [0.8574719823338359, 0.84830860693986, 0.8481889623892922, 0.8459374718079546, 0.8431100830503371, 0.8384370365131941, 0.8298279198056093, 0.8258933077667647, 0.8255168626652902, 0.8226315099866939, 0.8190106523801491, 0.8180207756789076, 0.8154023571223581, 0.8145186273191505, 0.8084945654723721, 0.8051729496797475, 0.7960830888896204, 0.7945010194721749, 0.777891565113962, 0.7720928155433907, 0.7675675783663938, 0.7566855241626946, 0.7519802752273343, 0.7433303658881685, 0.7261196264697187, 0.7143327944870622, 0.6574300327475657, 0.6307777369759111, 0.6260141643741666, 0.6199901785712558, 0.6042517081715322, 0.4953664247932583, 0.42354502459056353, 0.03583218257767578, 0.003379091517095373, 0, -0.032705788479513]
config_results['all_configs']['Decal']   = [0.9025676144841348, 0.8808973430585367, 0.8632569352297236, 0.843003836252914, 0.8412779549359748, 0.8326148577218725, 0.8259903603552909, 0.8251647362601252, 0.8234975267920435, 0.8131904305806954, 0.8026790847670227, 0.8020822305997628, 0.7993100667595154, 0.773299911255742, 0.7605103945768115, 0.7548175958841415, 0.7542600473811754, 0.7539624884608972, 0.7506961390606107, 0.7436162908286057, 0.7392560924984869, 0.7366408613629855, 0.7209907817896295, 0.6960887574173191, 0.6901768819523182, 0.6867081242471511, 0.6822679742233027, 0.6754355313436348, 0.6717918515047165, 0.019970797183754878, 0.01208553203193995, 0.0014424739801088523, 0, 0.0, -4.455235835787613e-05, -0.006558373642862577, -0.01965721679402126]
config_results['all_configs']['Hammer']  = [0.804030250628507, 0.7930056105016201, 0.792046312168701, 0.7775987067175913, 0.7723517999669374, 0.7691526495514234, 0.7653859825215631, 0.7580926173535748, 0.7382155098749168, 0.7368824810121988, 0.7291916989662692, 0.7275832032059261, 0.7130709090248033, 0.7034904383483079, 0.6996411123615431, 0.6984020417749214, 0.6792944374287432, 0.6746207843286218, 0.6659511091701124, 0.6596405034070918, 0.6511217486124348, 0.6510380160695421, 0.6460024225475129, 0.6439048863484191, 0.636763231652871, 0.6259795595624983, 0.6227427063450826, 0.5508929087729029, 0.5138475455858589, 0.4269002427374696, 0.3859610174870119, 0.3803954918546315, 0.2753250490711926, 0.1841630299733677, 0.04139696856474769, 0, -0.02905298129169586]
config_results['all_configs']['Munster'] = [0.959687503306432, 0.9596388461037777, 0.956956896082154, 0.9556404418474276, 0.9554407809606841, 0.9552539968216457, 0.9539838394183027, 0.9535180589596083, 0.9533703548987427, 0.9533462175258, 0.9499623579601896, 0.949074111280131, 0.9487545899393052, 0.9487191752383355, 0.9464038821121719, 0.9440980987957888, 0.9427060087009058, 0.9395373482236292, 0.9394163413709233, 0.9391573289398356, 0.9378997126637965, 0.9303702042941683, 0.9297515169008259, 0.9288284395655821, 0.9285985777600562, 0.9278140954107953, 0.8992949779883495, 0.888884656261457, 0.8781802419741398, 0.8345954935011288, 0.8196998160357126, 0.7774975234606374, 0.7485264484191585, 0.16504497358322, 0.1427715446998339, 0, 0.0]
config_results['all_configs']['Pedro']   = [0.6073451464510375, 0.603311991299097, 0.5771200256285797, 0.5709105002514614, 0.5669996108396216, 0.558619837122492, 0.5452100763141133, 0.5209169097303916, 0.5189296421970937, 0.5171709720118635, 0.5074673598534009, 0.5045296947723734, 0.4922800524481222, 0.45770926393610556, 0.45081932976580386, 0.4465092848714425, 0.4464220836158118, 0.4377866428519241, 0.43253394988158267, 0.4265284432271875, 0.4111062509017694, 0.40802164751707504, 0.37763130454952487, 0.36171543179676574, 0.3540424154772556, 0.3531117234349504, 0.3324829835653332, 0.3324709226260387, 0.17841923467877124, 0.06692416905153832, 0.04778185970605726, 0.01902193759692205, 0.014467216857024968, 0.014107059956596637, 0.0029921365486063624, 0, -0.0015806215318010096]
config_results['all_configs']['Kraut']   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
config_results['all_configs']['Katze']   = [0.8616133431165722, 0.8607821219396901, 0.8572496925197957, 0.8564293283035073, 0.8543191258277396, 0.85414678826315, 0.8503413783820133, 0.8393373625594724, 0.8392059291384224, 0.838460927779554, 0.8381813684965818, 0.8305470944500314, 0.8303379229309533, 0.8257757226943421, 0.8235324211892344, 0.822639363154757, 0.8218017202197803, 0.8196531123875234, 0.8170471211677353, 0.8109824693761103, 0.8090534115689513, 0.8053138645204736, 0.7955501700209058, 0.7952554832049057, 0.782583365729502, 0.7747769355666627, 0.7569330805093285, 0.7255328188855268, 0.6863542361506336, 0.6705793418225149, 0.6540542193703885, 0.5560962278055049, 0.4743939397910969, 0.101423477127349, 0.1005072017692087, 0, 0.0]
config_results['all_configs']['Kreatur'] = [0.8554919708387797, 0.8496014745001329, 0.8493688579474179, 0.8456579059883087, 0.8454138426372891, 0.8409002799782199, 0.8406446101413663, 0.8382464956569077, 0.8324944222153359, 0.832220610926065, 0.8288552406626706, 0.8280117451979119, 0.8241788851349063, 0.8204472858604969, 0.8167076942021294, 0.8162535078954828, 0.8130924769938654, 0.8108209257627674, 0.8094861689758412, 0.8079651982392373, 0.8077979821687586, 0.8066358977104926, 0.8042946005386382, 0.7977763341157843, 0.7696853703027509, 0.7678522019854566, 0.765870464585285, 0.7600940869722959, 0.7023102289418899, 0.6803191385433998, 0.6334904489640548, 0.42288133775404296, 0.42219870497075485, 0.14450197685147148, 0.014440745094815834, 0, 0.0]

config_results['same_config'] = {}
config_results['same_config']['Chucky']  = 0.8431100830503371
config_results['same_config']['Decal']   = 0.843003836252914
config_results['same_config']['Hammer']  = 0.804030250628507
config_results['same_config']['Munster'] = 0.16504497358322
config_results['same_config']['Pedro']   = 0.6073451464510375
config_results['same_config']['Kraut']   = 0
config_results['same_config']['Katze']   = 0.8170471211677353
config_results['same_config']['Kreatur'] = 0.7600940869722959


batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

datasets = ['Chucky', 'Decal', 'Hammer', 'Katze', 'Kraut', 'Kreatur', 'Munster', 'Pedro']
approaches = [('kakaobrain', '#0080FF'),
              ('precalc_64', '#FFB000'),
              ('precalc', '#FF0000')]


if __name__ == "__main__":
    for dataset in datasets:
        plt.figure(figsize=(4, 3))

        # plot individual approaches for different batch sizes
        for elem in approaches:
            label = elem[0]
            color = elem[1]
            plt.plot(config_results[label][dataset], lw=2, color=color, label=label)

        # plot results for the
        for elem in config_results['all_configs'][dataset]:
            plt.plot([elem]*9, lw=1, color='#00D000', alpha=0.1)

        plt.plot([config_results['same_config'][dataset]]*9, lw=2, color='#00D000', label='same_config')

        plt.legend(loc='lower right')
        plt.xlabel('batch size')
        plt.ylabel('ALC')
        plt.title(dataset)
        plt.ylim(-0.1, 1)
        plt.xticks(np.arange(9), batch_sizes)
        plt.savefig(dataset + '.svg', format='svg')
        plt.show()
