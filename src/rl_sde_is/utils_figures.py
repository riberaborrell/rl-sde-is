import matplotlib.pyplot as plt

COLORS_TAB10 = [plt.cm.tab10(i) for i in range(20)]

COLORS_TAB20 = [plt.cm.tab20(i) for i in range(20)]

COLORS_TAB20b = [plt.cm.tab20b(i) for i in range(20)]

COLORS_TAB20c = [plt.cm.tab20c(i) for i in range(20)]

COLORS_FIG = {
    'hjb': 'black',
}

TITLES_FIG = {
    'potential': r'Potential $V_\alpha$',
    'policy': r'Deterministic Policy $\mu_\theta$',
    'value-function': r'$V$',
    'u-l2-error': 'Estimation of $L^2(\mu)$',
    'loss': r'$\widehat{J}(\mu_\theta)$',
    'returns': r'Return $G_0(\tau)$',
    'time-steps': r'TS',
    'ct': r'CT(s)',
}
