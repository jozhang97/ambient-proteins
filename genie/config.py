from ast import literal_eval
from collections import OrderedDict

int_or_none = lambda x: int(x) if x is not None else None
float_or_none = lambda x: float(x) if x is not None else None
str_list_or_none = lambda x: x.strip().split(',') if x is not None else None
int_list_or_none = lambda x: [int(y) for y in x.strip().split(',')] if x is not None else None
eval_if_str = lambda x: literal_eval(x) if isinstance(x, str) else x

class Config:

    def __init__(self, filename=None):
        config = {} if filename is None else self._load_config(filename)
        self._create_config(config)

    def _create_config(self, config):

        self.io = {

            'name':                               config.pop('name',                                    None),
            'rootdir':                            config.pop('rootDirectory',                           'runs'),
            'datadir':                            config.pop('dataDirectory',                           'data/afdbreps_l-256_plddt_80/pdbs'),
            'min_n_res':              int_or_none(config.pop('minimumNumResidues',                      20)),
            'max_n_res':              int_or_none(config.pop('maximumNumResidues',                      256)),
            'max_n_chain':            int_or_none(config.pop('maximumNumChains',                        1)),
            'validation_split':     float_or_none(config.pop('validationSplit',                         None)),

            # Motif conditioning
            'motif_prob':                   float(config.pop('motifProbability',                        0.8)),
            'motif_min_pct_res':            float(config.pop('motifMinimumPercentageResidues',          0.05)),
            'motif_max_pct_res':            float(config.pop('motifMaximumPercentageResidues',          0.5)),
            'motif_min_n_seg':                int(config.pop('motifMinimumNumberSegments',              1)),
            'motif_max_n_seg':                int(config.pop('motifMaximumNumberSegments',              4)),

            "min_plddt":              int_or_none(config.pop("min_plddt",                               70)),
        }

        self.diffusion = {
            'n_timestep':                     int(config.pop('numTimesteps',                            1000)),
            'schedule':                           config.pop('schedule',                                'cosine'),

            "ambient":                           config.pop("ambient",                                  True),
            "dt_buffer":                     int(config.pop("dt_buffer",                                50)),
            "t_nature":                      int(config.pop("t_nature",                                 300)),
            "plddt_to_timestep":         OrderedDict(eval_if_str(config.pop("plddt_to_timestep",        ((90, -50), (80, 300), (70, 600))))),
            "always_ambient_for_high_noises":    config.pop("always_ambient_for_high_noises",           True),
        }

        self.model = {

            # General
            'c_s':                            int(config.pop('singleFeatureDimension',                  384)),
            'c_p':                            int(config.pop('pairFeatureDimension',                    128)),
            'rescale':                      float(config.pop('rescale',                                 1)),

            # Single feature network
            'c_pos_emb':                      int(config.pop('positionalEmbeddingDimension',            256)),
            'c_chain_emb':                    int(config.pop('chainEmbeddingDimension',                 64)),
            'c_timestep_emb':                 int(config.pop('timestepEmbeddingDimension',              512)),

            # Pair feature network
            'relpos_k':                       int(config.pop('relativePositionK',                       32)),
            'template_dist_min':            float(config.pop('templateDistanceMinimum',                 2)),
            'template_dist_step':           float(config.pop('templateDistanceStep',                    0.5)),
            'template_dist_n_bin':            int(config.pop('templateDistanceNumBins',                 37)),

            # Pair transform network
            'n_pair_transform_layer':         int(config.pop('numPairTransformLayers',                  5)),
            'include_mul_update':                 config.pop('includeTriangularMultiplicativeUpdate',   True),
            'include_tri_att':                    config.pop('includeTriangularAttention',              False),
            'c_hidden_mul':                   int(config.pop('triangularMultiplicativeHiddenDimension', 128)),
            'c_hidden_tri_att':               int(config.pop('triangularAttentionHiddenDimension',      32)),
            'n_head_tri':                     int(config.pop('triangularAttentionNumHeads',             4)),
            'tri_dropout':                  float(config.pop('triangularDropout',                       0.25)),
            'pair_transition_n':              int(config.pop('pairTransitionN',                         4)),

            # Structure network
            'n_structure_layer':              int(config.pop('numStructureLayers',                      8)),
            'n_structure_block':              int(config.pop('numStructureBlocks',                      1)),
            'c_hidden_ipa':                   int(config.pop('ipaHiddenDimension',                      16)),
            'n_head_ipa':                     int(config.pop('ipaNumHeads',                             12)),
            'n_qk_point':                     int(config.pop('ipaNumQkPoints',                          4)),
            'n_v_point':                      int(config.pop('ipaNumVPoints',                           8)),
            'ipa_dropout':                  float(config.pop('ipaDropout',                              0.1)),
            'n_structure_transition_layer':   int(config.pop('numStructureTransitionLayers',            1)),
            'structure_transition_dropout': float(config.pop('structureTransitionDropout',              0.1)),

        }

        self.training = {
            'seed':                           int(config.pop('seed',                                    100)),
            'n_epoch':                        int(config.pop('numEpoches',                              1)),
            'batch_size':                     int(config.pop('batchSize',                               1)),
            'log_every_n_step':               int(config.pop('logEverySteps',                           1000)),
            'checkpoint_every_n_epoch':       int(config.pop('checkpointEveryEpoches',                  500)),
            'condition_loss_weight':          int(config.pop('conditionLossWeight',                     1)),
            'resume':                             config.pop("resume",                                  None),
            'accum_grad':                       int(config.pop('accumGrad',                             1)),
        }

        self.optimization = {
            "warmup_epochs":                  int(config.pop("warmupEpoches",                           5)),
            "weight_decay":                 float(config.pop("weightDecay",                             1e-4)),
            "use_lr_schedule":                    config.pop("useLRSchedule",                           True),
            'lr':                           float(config.pop('learningRate',                            1e-4))
        }

        # Print the keys in the input config that are not being used in this function
        if config:
            print("[WARNING] not all config keys are being used")
            for k, v in config.items():
                print(f"  - {k}: {v}")

    def _load_config(self, filename):
        config = {}
        with open(filename) as file:
            for line in file:
                elts = line.split()
                assert len(elts) in [0, 2], f"Invalid config file: {filename}, line: {line}"
                if len(elts) == 2:
                    if elts[1] == 'True':
                        config[elts[0]] = True
                    elif elts[1] == 'False':
                        config[elts[0]] = False
                    else:
                        config[elts[0]] = elts[1]
        return config
