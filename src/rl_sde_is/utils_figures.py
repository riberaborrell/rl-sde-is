import matplotlib.pyplot as plt

COLORS_TAB10 = [plt.cm.tab10(i) for i in range(20)]

COLORS_TAB20 = [plt.cm.tab20(i) for i in range(20)]

COLORS_TAB20b = [plt.cm.tab20b(i) for i in range(20)]

COLORS_TAB20c = [plt.cm.tab20c(i) for i in range(20)]

COLORS_FIG = {
    'hjb': 'black',
}

TITLES_FIG = {
    'potential': r'Potential $V_\alpha(s)$',
    'policy': r'Deterministic policy $\mu_\theta(s)$',
    'value-function': r'Value function $V_\omega(s)$',
    'q-value-function': r'Q-value function $Q_\omega(s, a)$',
    'a-value-function': r'Advantage function $A_\omega(s, a)$',
    'value-rms-error': r'RMS Error of $V(s)$',
    'policy-rms-error': r'RMS Error of $\pi(s)$',
    'policy-l2-error': r'Estimation of $L^2(\mu_\theta)$',
    'loss': r'$\widehat{J}(\mu_\theta)$',
    'returns': r'Return $G_0(\tau)$',
    'time-steps': r'TS',
    'ct': r'CT(s)',
}
