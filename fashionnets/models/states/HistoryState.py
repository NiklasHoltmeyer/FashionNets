import collections
import os
import pickle
from collections import defaultdict

from fashionscrapper.utils.list import distinct, flatten
from tensorflow.keras import backend


class HistoryState:
    def __init__(self, history, params=None, history_history=None, epoch=None):
        if history is not None:
            self.params = backend.get_value(history.params)
            self.history_history = backend.get_value(history.history)
            self.epoch = backend.get_value(history.epoch)
        else:
            self.params = params
            self.history_history = history_history
            self.epoch = epoch

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path, skip_assert=False):
        if not path.endswith(".pkl"):
            params, epochs, history = HistoryState.load_histories(path, skip_assert=skip_assert)
            return HistoryState(history=None, params=params, history_history=history, epoch=epochs)
        with open(path, 'rb') as f:
            state = pickle.load(f)
            return state

    @staticmethod
    def load_histories(path, skip_assert=False):
        files = os.listdir(path)
        history_file_paths = filter(lambda f: "history" in f, files)
        history_file_paths = filter(lambda f: f.endswith(".pkl"), history_file_paths)
        history_file_paths = map(lambda f: os.path.join(path, f), history_file_paths)

        history_files = map(lambda p: HistoryState.load(p, skip_assert=skip_assert), history_file_paths)
        history_files = list(history_files)

        if len(history_files) < 1:
            raise Exception("Could not Find History File! (File must Contain 'history' and End with '.pkl')")

        if len(history_files) == 1:
            return history_files[0]

        history_files = sorted(history_files, key=lambda o: o.epoch[-1])
        history_files = list(history_files)

        latest_history = history_files[-1]

        params = latest_history.params
        values_by_ep_metric = defaultdict(lambda: defaultdict(lambda: []))

        def flatten_history_epoch(history):
            d = {}
            for loss_name, loss_values in history.history_history.items():
                d[loss_name] = {}

                for idx, loss in enumerate(loss_values):
                    ep = history.epoch[idx]
                    d[loss_name][str(ep)] = loss
            data = []
            for ep in history.epoch:
                ep_data = {}
                for loss in history.history_history.keys():
                    ep_data[loss] = d[loss][str(ep)]
                data.append((ep, ep_data))

            return data

        epoch_values = map(flatten_history_epoch, history_files)
        epoch_values = flatten(epoch_values)

        for epoch, values in epoch_values:
            for metric, metric_values in values.items():
                assert not isinstance(metric_values, list)
                values_by_ep_metric[epoch][metric].append(metric_values)

        values_by_ep_metric = dict(collections.OrderedDict(sorted(values_by_ep_metric.items())))
        values_by_ep_metric = {k: dict(v) for k, v in values_by_ep_metric.items()}

        epochs = []
        history = defaultdict(lambda: [])
        for ep, values in values_by_ep_metric.items():
            epochs.append(ep)
            for metric, metric_values in values.items():
                metric_values = distinct(metric_values)

#                assert len(metric_values) ==1, f"len(metric_values) = {len(metric_values)}, " \
#                         f"Path = {path}, metric_values = {metric_values}, Epoch = {ep}"
                if len(metric_values) == 1:
                    metric_value = metric_values[0]
                else:
                    print(f"Warning Averaging {len(metric_values)} Values to One Value")
                    metric_value = sum(metric_values) / len(metric_values)
                history[metric].append(metric_value)
        number_elements = distinct([len(x) for x in history.values()])

        if not skip_assert:
            assert len(number_elements) == 1, "ALl Metric Values should have the same Number of Values!"
            assert number_elements[0] == len(epochs)

        return params, epochs, dict(history)

    def apply(self, model):
        model.history.params = self.params
        model.history.history = self.history_history
        model.history.epoch = self.epoch

    def __eq__(self, other):
        if not isinstance(other, HistoryState):
            return False

        for attribute in self.__dict__.keys():
            if getattr(self, attribute) != getattr(other, attribute):
                return False

        return True

    def __str__(self):
        lines = ["{"]
        for k, v in self.__dict__.items():
            line = f"\t{k}:\t{v},"
            lines.append(line)
        lines.append("}")
        return "\n".join(lines)


if __name__ == "__main__":
##    # path = r"D:\Download\311_resnet50_imagenet_quadruplet_history-0004.pkl"
##    # print(HistoryState.load(path))
##    path = r"D:\masterarbeit_runs\11_quadruplet_apn_q\history_files"
##    #path = r"D:\masterarbeit_runs\11_quadruplet_apn_q\history_files\11_quadruplet_apn_q_history-0026.pkl"
##    print(HistoryState.load(path))
    loss = [29.05521011, 3.826568127, 1.740655303, 1.004055381, 0.799381435, 0.7429474, 0.750654936, 0.75475502,
            0.744941592, 0.792005122, 0.773509443, 0.726489902, 0.707795918, 0.685467958, 0.634136677, 0.596520126,
            0.596342623, 0.591294289, 0.5828228, 0.558324516, 0.542056024, 0.515808284, 0.493434936, 0.486381918,
            0.466014534, 0.472183704, 0.455131441, 0.432596505, 0.44989419, 0.443586469, 0.42249161, 0.411863923,
            0.417138964, 0.395416051, 0.384879172, 0.382404059, 0.373667806, 0.376167059, 0.353913635, 0.354777902,
            0.363204062, 0.363265187, 0.360880971, 0.350860029, 0.349808574, 0.336135209, 0.322923422, 0.315965503,
            0.319143414, 0.305973709, 0.308191806, 0.303754359, 0.313735247, 0.312202007, 0.301015884, 0.300245792,
            0.290976852, 0.292242169, 0.29128924, 0.296791226, 0.288228571, 0.2729415, 0.273308843, 0.277834773,
            0.264223069, 0.261469007, 0.266658634, 0.268006057, 0.249784157, 0.243626967, 0.243579656, 0.242969245,
            0.240301058, 0.235440627, 0.239027992, 0.234901577, 0.237481683, 0.236132294, 0.229417965, 0.228795812,
            0.22797592, 0.219396219, 0.220393106, 0.220944479, 0.21576032, 0.213267297, 0.208731726, 0.206867769,
            0.207281247, 0.206549153, 0.207552224, 0.204038545, 0.202568427, 0.198047906, 0.200882286, 0.203448772,
            0.199248716, 0.19546704, 0.193929166, 0.186419696, 0.188703805, 0.178932086, 0.186294839, 0.185641035,
            0.184212759, 0.182455167, 0.185525268, 0.185060978, 0.186544016, 0.181365788, 0.181752026, 0.175559029,
            0.172784135, 0.174256295, 0.171500385, 0.174467921, 0.175537601, 0.172411397, 0.169835597, 0.172140896,
            0.174906105, 0.163481608, 0.163237229, 0.164202198, 0.16741325, 0.165263847, 0.1604902, 0.163343295,
            0.167735279, 0.161696896, 0.16104646, 0.16113776, 0.158858418, 0.160165563, 0.154949635, 0.159873024,
            0.159242168, 0.156692863, 0.157830372, 0.159699202, 0.156693801, 0.158565789, 0.156561673, 0.156173766,
            0.149615929, 0.148504034, 0.155639976, 0.152098969, 0.156436205, 0.153012708]
    val_loss = [19.7319603, 3.076832294, 1.317795634, 0.833470047, 0.654936254, 0.769790471, 0.756762087, 0.728375673,
                0.857274234, 0.78134793, 0.765329838, 0.718318224, 0.694375217, 0.662402093, 0.657944381, 0.568202674,
                0.781124055, 0.588325858, 0.571516037, 0.543953001, 0.536181629, 0.49564153, 0.547873318, 0.550867617,
                0.487232864, 0.475877792, 0.552762389, 0.454993427, 0.482903391, 0.523057282, 0.454589695, 0.447820157,
                0.438219905, 0.418537825, 0.417877704, 0.427908063, 0.417092741, 0.412105531, 0.414370835, 0.370036662,
                0.390382856, 0.429943055, 0.402725458, 0.413155466, 0.42633009, 0.423644066, 0.396641314, 0.36240077,
                0.368294358, 0.368419439, 0.347536236, 0.370637894, 0.450061083, 0.37540552, 0.371662289, 0.373409241,
                0.363600641, 0.361015618, 0.361921668, 0.364761502, 0.379533887, 0.377895385, 0.338878036, 0.38859567,
                0.332900763, 0.364675164, 0.349163473, 0.360889286, 0.341992319, 0.330498427, 0.327794701, 0.322779238,
                0.350658506, 0.318790078, 0.333586365, 0.344944924, 0.332783252, 0.361813635, 0.341514349, 0.321814597,
                0.328664273, 0.313683718, 0.33259505, 0.326572537, 0.305998832, 0.310898334, 0.344626307, 0.314900249,
                0.337194026, 0.334145933, 0.318550795, 0.327637911, 0.315334916, 0.310639381, 0.310949057, 0.334948182,
                0.320834517, 0.302247375, 0.313623548, 0.307266623, 0.337446183, 0.30474475, 0.299662173, 0.311540961,
                0.308960795, 0.325871378, 0.314679086, 0.304604799, 0.340341747, 0.328124315, 0.301890492, 0.299149334,
                0.29442364, 0.296415716, 0.304831862, 0.317917079, 0.303341687, 0.300606996, 0.319063634, 0.3083812,
                0.335760713, 0.334714413, 0.313757479, 0.313884914, 0.288475752, 0.310300678, 0.307774395, 0.29357931,
                0.309145063, 0.328923702, 0.306491345, 0.316668957, 0.325793266, 0.294843376, 0.295046777, 0.301254958,
                0.335438401, 0.30229634, 0.3110632, 0.312537044, 0.322711021, 0.321374774, 0.316875547, 0.322809845,
                0.340673923, 0.287813365, 0.303002149, 0.305033833, 0.336776227, 0.298158616, ]
    epochs = list(range(len(val_loss)))
    path = r"D:\masterarbeit_runs\231_triplet_ctl_t\history_files"

    state = HistoryState.load(path)
    print(state.history_history["val_loss"])
    #self.params = params
    # self.history_history = history_history
    #self.epoch = epoch





