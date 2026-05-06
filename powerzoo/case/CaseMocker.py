import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

np.set_printoptions(precision=4, suppress=True)
plt.rcParams["font.sans-serif"] = ["SimSun"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# Support Chinese characters
# plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
# plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly


class CostBase():
    def __init__(self, pmax, a, b, c):
        self.pmax = pmax
        self.a = a
        self.b = b
        self.c = c


cb100 = CostBase(100, 0.027, -3.6, 550)
cb100_1 = CostBase(100, 0.04, -3.6, 550)
cb100_2 = CostBase(100, 0.04, -2.5, 550)
cb100_3 = CostBase(100, 0.15, -10, 550)
cb100_4 = CostBase(100, 0.046257 * 3, -7.5863 * 2, 512.61)
cb100_coal = CostBase(100, 0.025, -2.1015, 71.0)
cb100_gas = CostBase(100, 0.039551, -4.3164, 141.1)
cb100_nuclear = CostBase(100, 0.008654, -1.0638, 34.44)
cb_dict = {0: cb100, 1: cb100_1, 2: cb100_2, 3: cb100_3, 4: cb100_4, 5: [cb100_coal, cb100_coal, cb100_coal, cb100_gas, cb100_gas, cb100_nuclear]}


def truncnorm(size=1, mu=0.9, sigma=0.1, lower=0.65, upper=1.0):
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(size)


class CaseMocker():
    def __init__(self, c, cb=cb100_coal, pmin=0.3, real_mc=False):
        self.c = c
        self.real_params = getattr(self.c, 'real_params', False)
        pmax = self.pmax = c.units['p_max'].values
        start_up_cost_bin = np.array([[150, 200000], [300, 400000], [600, 600000], [1000, 800000]])

        if self.real_params:
            self.mc_a = self.c.units['mc_a']
            self.mc_b = self.c.units['mc_b']
            self.mc_c = self.c.units['mc_c']
            self.ac_a = self.mc_a / 3
            self.ac_b = self.mc_b / 2
            self.ac_c = self.mc_c
            self.pmin = self.c.units['p_min'] / self.c.units['p_max']
            pmin_power = self.c.units['p_min']
            if 'init_no_load_cost' in self.c.units:
                self.no_load_cost = self.c.units['init_no_load_cost']
            else:
                self.no_load_cost = self.ac_a * pmin_power ** 3 + self.ac_b * pmin_power ** 2 + self.ac_c * pmin_power
            if 'init_start_up_cost' in self.c.units:
                self.start_up_cost = self.c.units['init_start_up_cost']
            else:
                start_up_cost_tile = np.tile(start_up_cost_bin[:, 0], (len(pmax), 1)).T - pmax
                self.start_up_cost = (np.where(start_up_cost_tile >= 0, 1, 100) * start_up_cost_bin[:, [1]]).min(0)

        else:
            p_k = cb.pmax / pmax
            self.mc_a = cb.a * p_k * p_k
            self.mc_b = cb.b * p_k
            self.mc_c = cb.c if real_mc else (cb.c * 50 / (pmax / cb.pmax + 50)).round(0)  # Apply scaling so large units have lower cost; 50 is manually chosen
            self.ac_a = self.mc_a / 3
            self.ac_b = self.mc_b / 2
            self.ac_c = self.mc_c
            self.pmin = pmin
            pmin_power = pmin * pmax
            self.no_load_cost = self.ac_a * pmin_power ** 3 + self.ac_b * pmin_power ** 2 + self.ac_c * pmin_power
            start_up_cost_tile = np.tile(start_up_cost_bin[:, 0], (len(pmax), 1)).T - pmax
            self.start_up_cost = (np.where(start_up_cost_tile >= 0, 1, 100) * start_up_cost_bin[:, [1]]).min(0)

    def mc(self, p):
        return self.mc_a * p ** 2 + self.mc_b * p + self.mc_c

    def mock_c(self):
        if self.real_params:
            return self, self.c
        self.c.units['mc_a'] = self.mc_a
        self.c.units['mc_b'] = self.mc_b
        self.c.units['mc_c'] = self.mc_c
        self.c.units['p_min'] = self.pmin * self.c.units['p_max'].values
        # self.c.units['approved'] = self.mc_c - 100
        cost = self.c.units.loc[:, ['mc_a', 'mc_b', 'mc_c']]
        self.c.units['approved'] = (cost['mc_a'].values * self.pmax ** 2) / 3 + (cost['mc_b'].values * self.pmax) / 2 + cost['mc_c'].values

        self.c.units['Mi'] = 0.1
        self.c.units['real_no_load_cost'] = self.no_load_cost
        self.c.units['real_start_up_cost'] = self.start_up_cost

        return self, self.c

    def mock_band(self, band_num):
        min_mc_x = -self.mc_b / self.mc_a / 2
        amounts = []
        prices = []
        amounts.append(min_mc_x / self.pmax - self.pmin)
        prices.append(self.mc(min_mc_x))
        other_bands = (1 - min_mc_x / self.pmax) / (band_num - 1)
        for i in range(band_num - 1):
            amounts.append(other_bands)
            prices.append(self.mc(min_mc_x + (i + 1) * other_bands * self.pmax))
        # amounts = np.ones_like(np.array(amounts))
        return np.array(amounts).T, np.array(prices).T

    def random_price(self, band_num):
        amounts, prices = self.mock_band(band_num)
        prices_T = prices.T
        num = prices_T.shape[1]
        p = np.zeros(num)
        for i in range(band_num):
            p_ = truncnorm(size=num, mu=prices_T[i], sigma=0.1 * prices_T[i], lower=p[-1], upper=2 * prices_T[i])
            p = np.vstack((p, p_))
        return amounts, p[1:].T

    def truncated_normal_array(self, n=96, mu=1.0, var=0.003, low=0.8, high=1.05, seed=41):
        """
        Sample n values from a truncated normal distribution over [low, high] with base N(mu, var).
        Note: After truncation, the empirical mean/variance differ from mu/var; mu/var are for the untruncated base distribution.
        """
        sigma = np.sqrt(var)
        a = (low - mu) / sigma
        b = (high - mu) / sigma
        rng = np.random.default_rng(seed)
        samples = stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)
        return samples

    def upsample_24_to_n(self, loads, target_periods=96):
        """
        Resample a 24-hour load curve to the specified number of periods
        
        Args:
            loads: shape (batch, 24) - 24-hour load curve
            target_periods: int - target periods, e.g., 48 (half-hour), 96 (15 min), 288 (5 min)
            
        Returns:
            shape (batch, target_periods) - resampled load curve
        """
        if loads.ndim != 2 or loads.shape[1] != 24:
            raise ValueError("loads must be a 2-D array with 24 columns")

        cols = loads.shape[1]  # 24
        x_new = np.linspace(0, cols - 1, target_periods)  # 0,...,23, length is target_periods

        # Compute left/right indices and weight t for linear interpolation
        j0 = np.floor(x_new).astype(int)  # Left endpoint index
        j1 = np.clip(j0 + 1, 0, cols - 1)  # Right endpoint index (last point equals itself)
        t = (x_new - j0)[None, :]  # Broadcast to each row, shape (1, target_periods)

        v0 = loads[:, j0]  # Left endpoint values, shape (N, target_periods)
        v1 = loads[:, j1]  # Right endpoint values, shape (N, target_periods)

        loads_resampled = (1 - t) * v0 + t * v1  # Linear interpolation
        return loads_resampled

    def mock_load(self, periods_num, random=False, no_mock=False, mu=0.9, sigma=0.1, lower=0.65, upper=1.0):
        """
        Generate load curves according to the specified number of periods
        
        Args:
            periods_num: int - target periods, e.g., 24 (hour), 48 (half-hour), 96 (15 min), 288 (5 min)
            random: bool - whether to add random perturbation
            no_mock: bool - whether to skip using the load-curve template
            mu, sigma, lower, upper: float - parameters for random perturbation
            
        Returns:
            shape (batch, periods_num) - generated load curves
        """
        data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'load_curve.csv'), delimiter=",")
        loads = self.c.loads['d_max']
        if no_mock:
            loads_curve = np.array(loads)[:, np.newaxis] * np.ones(24)
        else:
            if random:
                loads_curve = np.array(loads)[:, np.newaxis] * data * truncnorm(24, mu, sigma, lower, upper)  # Truncated normal factors
            else:
                loads_curve = np.array(loads)[:, np.newaxis] * data

        # Handle periods_num cases
        if periods_num <= 24:
            # If target periods <= 24, take the first periods_num columns directly
            return loads_curve[:, :periods_num]
        else:
            # For periods > 24, use the resampling function
            return self.upsample_24_to_n(loads_curve, target_periods=periods_num)

    # @staticmethod
    def draw_cost(self, path, amounts, prices, offer_no_load_cost, c=None, with_cost=True, width="100%", height="400", grid=False):
        import pyecharts.options as opts
        from pyecharts.charts import Line
        from pyecharts.charts import Grid, Page

        c = self.c if c is None else c
        unit_amounts_cumsum = np.dot(np.diag(c.units.loc[:, 'p_max']), amounts.cumsum(1)) + c.units.loc[:, ['p_min']].values
        real_no_load_cost = c.units['real_no_load_cost'].values
        unit_min = c.units['p_min']

        l0 = Line(init_opts=opts.InitOpts(width=f"{str(width).replace('px', '')}px", height=f"{str(height).replace('px', '')}px"), )
        l_list = []
        if grid:
            for _ in range(c.units.shape[0]):
                _l = Line(init_opts=opts.InitOpts(width=f"{str(width).replace('px', '')}px", height=f"{str(height).replace('px', '')}px"), )
                l_list.append(_l)

        for i, (amount, price, pmin, noload) in enumerate(
                zip(unit_amounts_cumsum.round(2), prices.round(2), unit_min, (offer_no_load_cost / unit_min).round(2))):
            l = l_list[i] if grid else l0
            l = l.add_xaxis([pmin, *amount]).add_yaxis(f'Unit{i + 1}@{c.units.iloc[i]["bus_id"].astype(int)}', [0, *price], is_step=True,
                                                       linestyle_opts=opts.LineStyleOpts(width=3),
                                                       label_opts=opts.LabelOpts(position='bottom')). \
                add_xaxis([0, pmin, pmin]).add_yaxis(f'Unit{i + 1}@{c.units.iloc[i]["bus_id"].astype(int)}', [noload, noload, 0], is_step=True,
                                                     linestyle_opts=opts.LineStyleOpts(width=3), )

        if with_cost:
            cost = c.units.loc[:, ['mc_a', 'mc_b', 'mc_c']]
            xs = np.linspace(0, c.units['p_max'], 50)
            ys = cost['mc_a'].values * xs ** 2 + cost['mc_b'].values * xs + cost['mc_c'].values
            for i, (x, y) in enumerate(zip(xs.round(2).T, ys.round(2).T)):  # energy cost
                l = l_list[i] if grid else l0
                l = l.add_xaxis(x).add_yaxis(f'Unit{i + 1}@{c.units.iloc[i]["bus_id"].astype(int)}', y, label_opts=opts.LabelOpts(is_show=False),
                                             is_smooth=True, symbol="emptyCircle", is_symbol_show=False,
                                             linestyle_opts=opts.LineStyleOpts(width=2, type_='dotted'), )

            for i, (pmin, noload) in enumerate(zip(unit_min, (real_no_load_cost / unit_min).round(2))):  # no-load cost
                l = l_list[i] if grid else l0
                l = l.add_xaxis([0, pmin, pmin]).add_yaxis(f'Unit{i + 1}@{c.units.iloc[i]["bus_id"].astype(int)}', [noload, noload, 0], is_step=True,
                                                           linestyle_opts=opts.LineStyleOpts(width=2, type_='dashed'))

        if grid:
            for i, l in enumerate(l_list):
                l.set_global_opts(title_opts=opts.TitleOpts(title=f"Cost curve {i}"),
                                  # datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100),
                                  xaxis_opts=opts.AxisOpts(type_="value",
                                                           axistick_opts=opts.AxisTickOpts(is_show=True),
                                                           splitline_opts=opts.SplitLineOpts(is_show=True), ),
                                  yaxis_opts=opts.AxisOpts(type_="value",
                                                           axistick_opts=opts.AxisTickOpts(is_show=True),
                                                           splitline_opts=opts.SplitLineOpts(is_show=True), ),
                                  tooltip_opts=opts.TooltipOpts(trigger="axis"), )
        else:
            l0.set_global_opts(title_opts=opts.TitleOpts(title=f""),
                               # datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100),
                               xaxis_opts=opts.AxisOpts(type_="value",
                                                        axistick_opts=opts.AxisTickOpts(is_show=True),
                                                        splitline_opts=opts.SplitLineOpts(is_show=True), ),
                               yaxis_opts=opts.AxisOpts(type_="value",
                                                        axistick_opts=opts.AxisTickOpts(is_show=True),
                                                        splitline_opts=opts.SplitLineOpts(is_show=True), ),
                               tooltip_opts=opts.TooltipOpts(trigger="axis"), )
        if grid:
            page = Page()
            for l in l_list:
                page.add(l)
            page.render(path)
        else:
            l0.render(path)

    def draw_real_cost(self, cost_type='am', title=''):  # a: average cost, m: marginal cost
        units = self.c.units
        cols = 6
        rows = (len(units) - 1) // cols + 1
        fig, ax_list = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharey=True, dpi=200)
        for (i, unit), ax in zip(units.iterrows(), ax_list.flatten()):
            cost = unit.loc[['mc_a', 'mc_b', 'mc_c']]
            p_max = unit.loc['p_max']
            xs = np.linspace(0, p_max, 50)
            if 'm' in cost_type:
                ys = cost['mc_a'] * xs ** 2 + cost['mc_b'] * xs + cost['mc_c']
                ax.plot(xs, ys, c=colors[0], alpha=0.8, label='Marginal cost', ls='-')
            if 'a' in cost_type:
                ys_ac = cost['mc_a'] * xs ** 2 / 3 + cost['mc_b'] * xs / 2 + cost['mc_c']
                ax.plot(xs, ys_ac, c=colors[1], alpha=0.8, label='Average cost')
            if 'p' in cost_type:
                ys_ac = unit.loc['approved']
                ax.axhline(y=ys_ac, c="r", ls="--", lw=0.5, label='Approved price')

            p_min = unit.loc['p_min']
            # no_load_cost = unit.loc['real_no_load_cost'] / p_min
            xs_pmin = np.linspace(0, p_min, 50)
            ys_pmin = cost['mc_a'] * xs_pmin ** 2 + cost['mc_b'] * xs_pmin + cost['mc_c']
            ax.fill_between(xs_pmin, 0, ys_pmin, alpha=.2, linewidth=0, label='No-load cost')

            ax.legend(loc=4)
            # ax.set_ylim(ymin=200)
            ax.set_xlim(xmin=0)
            ax.set_xlabel('Capacity (MW)')
            ax.set_ylabel('Price (per MW)')
        plt.suptitle(title)
        fig.tight_layout()
        plt.show()
        self.showed = True
        return self

    def draw_single_unit_offer_real_cost(self, unit_id=3, cost_type='am', title=''):
        units = self.c.units
        unit = units.loc[unit_id]
        unit_sid = unit['#id'].astype(int)
        fig, ax_list = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True, dpi=200)

        cost = unit.loc[['mc_a', 'mc_b', 'mc_c']]
        p_min = unit.loc['p_min']
        p_max = unit.loc['p_max']
        p_real = 400

        xs = np.linspace(0, p_max, 50)
        for ax in ax_list:
            if 'm' in cost_type:
                ys = cost['mc_a'] * xs ** 2 + cost['mc_b'] * xs + cost['mc_c']
                ax.plot(xs, ys, c=colors[0], alpha=0.8, label='Marginal cost curve', ls='-')
            if 'a' in cost_type:
                ys_ac = cost['mc_a'] * xs ** 2 / 3 + cost['mc_b'] * xs / 2 + cost['mc_c']
                ax.plot(xs, ys_ac, c=colors[1], alpha=0.8, label='Average cost curve')
            ax.axvline(p_min, linestyle='--', c='grey', alpha=0.5)
            ax.axvline(p_real, linestyle='--', c='grey', alpha=0.5)
        ax = ax_list[0]

        # no_load_cost = unit.loc['real_no_load_cost'] / p_min
        xs_pmin = np.linspace(0, p_min, 50)
        ys_pmin = cost['mc_a'] * xs_pmin ** 2 + cost['mc_b'] * xs_pmin + cost['mc_c']
        ax.fill_between(xs_pmin, 0, ys_pmin, alpha=.3, facecolor=colors[4], linewidth=0, label='Actual no-load cost')

        xs_preal = np.linspace(p_min, p_real, 50)
        ys_preal = cost['mc_a'] * xs_preal ** 2 + cost['mc_b'] * xs_preal + cost['mc_c']
        ax.fill_between(xs_preal, 0, ys_preal, alpha=.3, linewidth=0, facecolor=colors[2], label='Actual energy cost')
        ax.set_ylabel('Price (per MW)')

        ax = ax_list[1]
        noload_p_min = unit.loc['real_no_load_cost'] / p_min + 10

        amounts, prices = self.mock_band(5)
        amounts_step = np.stack((p_min, p_min, *amounts[unit_sid].cumsum() * p_max + p_min))
        prices_step = np.stack((0, *prices[unit_sid], prices[unit_sid, -1])) + 10
        ax.step(amounts_step, prices_step, where='post', label='Energy offer curve', c=colors[2], ls='-', alpha=0.8)

        ax.plot([0, p_min, p_min], [noload_p_min, noload_p_min, 0], c=colors[4])
        xs_pmin = np.linspace(0, p_min, 50)
        ax.fill_between(xs_pmin, 0, noload_p_min, alpha=.3, facecolor=colors[4], linewidth=0, label='No-load offer cost')

        xs_preal = np.append(amounts_step[amounts_step < p_real][1:].repeat(2)[1:], p_real)
        ys_preal = prices_step[1:].repeat(2)[: len(xs_preal)]
        ax.fill_between(xs_preal, 0, ys_preal, alpha=.3, linewidth=0, facecolor=colors[2], label='Energy offer cost')

        for ax in ax_list:
            ax.legend(loc=4)
            ax.set_ylim(ymin=0)
            ax.set_xlim(xmin=0, xmax=p_max)
            ax.set_xlabel('Capacity (MW)')
        ax_list[0].set_title('(a) Actual unit cost (schematic)', loc='center', y=-0.27)
        ax_list[1].set_title('(b) Offer / bid cost (schematic)', loc='center', y=-0.27)
        # plt.suptitle(title)
        fig.tight_layout()
        # plt.savefig('figs/unit-real-and-offer-cost.png')
        plt.show()
        return self
