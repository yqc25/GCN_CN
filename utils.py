import torch
import sys
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import degree


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    valid_ap = average_precision_score(val_true, val_pred)
    test_ap = average_precision_score(test_true, test_pred)
    results = dict()
    results['AUC'] = (valid_auc, test_auc)
    results['AP'] = (valid_ap, test_ap)
    return results


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout, last_best=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            if last_best:
                # get last max value index by reversing result tensor
                argmax = result.size(0) - result[:, 0].flip(dims=[0]).argmax().item() - 1
            else:
                argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []

            for r in result:
                valid = r[:, 0].max().item()
                if last_best:
                    # get last max value index by reversing result tensor
                    argmax = r.size(0) - r[:, 0].flip(dims=[0]).argmax().item() - 1
                else:
                    argmax = r[:, 0].argmax().item()
                test = r[argmax, 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)