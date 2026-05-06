"""Power market case base module

This module contains basic data structures and computation methods for power market simulation.
"""
from typing import List, Union
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp


class DataFrame(pd.DataFrame):
    """Extended DataFrame for power market data display
    
    Inherits from pandas.DataFrame with label feature and custom display format.
    """

    def __init__(self, columns: List[str], data: Union[List, np.ndarray]):
        """Initialize DataFrame
        
        Args:
            columns: column names
            data: data content
        """
        super(DataFrame, self).__init__(data=data, columns=columns)
        self.loc[:, '#id'] = self.index
        self.index = self['id']
        self.label = ''

    def __repr__(self) -> str:
        """Custom string representation"""
        return f'{"=" * 20} {self.label} {"=" * 20}\n{self.to_string(index=False)}'

    def __str__(self) -> str:
        """String representation"""
        return self.__repr__()

    def set_label(self, label: str) -> 'DataFrame':
        """Set label
        
        Args:
            label: label name
            
        Returns:
            self to support method chaining
        """
        self.label = label
        return self


class ClearCase(object):
    """Base class for power market clearing cases
    
    Contains basic data for nodes, units, lines and loads,
    along with matrix computations and data processing methods.

    Subclasses should override the class-level metadata attributes below
    to enable automatic discovery, filtering, and compatibility validation.
    """

    GRID_TYPE: str = ""
    BUS_COUNT: int = 0
    PHASE: str = "1"
    VOLTAGE_LEVEL: str = ""
    SOURCE: str = ""
    DESCRIPTION: str = ""

    def __init__(self, mock=True):
        """Initialize clearing case"""
        self.init_flag = False
        self._node_gsdf = None
        self._node_ptdf = None
        self._plotter = None
        self.mocker = None
        self.nodes_map_units = None
        self.flexloads_num = len(getattr(self, 'flexloads', []))
        if mock:
            from .CaseMocker import CaseMocker
            self.mocker, _ = CaseMocker(self).mock_c()
            
        # if lines['cap'] == 0 or lines['floor'], set to 1000000
        self.lines.loc[self.lines['cap'] == 0, 'cap'] = 1000000
        self.lines.loc[self.lines['floor'] == 0, 'floor'] = -1000000

    def init(self) -> 'ClearCase':
        """Initialize case data
        
        Set node ID mapping and basic flags.
        
        Returns:
            self to support method chaining
        """
        self.init_flag = True
        self.units['#bus_id'], self.loads['#bus_id'] = self.get_nodes_id(
            self.units['bus_id'], self.loads['bus_id']
        )
        if self.flexloads_num > 0:
            # The trailing comma is required because the function returns a list object which needs ',' unpacking.
            self.flexloads['#bus_id'], = self.get_nodes_id(self.flexloads['bus_id'])
        self.lines['#from'], self.lines['#to'] = self.get_nodes_id(
            self.lines['from'], self.lines['to'])
        if not hasattr(self, 'name'):
            self.name = 'Case'
        return self

    def __repr__(self) -> str:
        """String representation"""
        return '\n'.join(self.to_str())

    # =============================================================================
    # Plotter property
    # =============================================================================

    @property
    def plotter(self):
        """Get the CasePlotter instance (lazy initialization)
        
        Returns:
            CasePlotter instance for visualizing this case
        """
        if self._plotter is None:
            from .CasePlotter import CasePlotter
            self._plotter = CasePlotter(self)
        return self._plotter

    # =============================================================================
    # Data validation methods
    # =============================================================================

    def check(self) -> None:
        """Check data integrity
        
        Verify that each table contains the required columns.
        
        Raises:
            AssertionError: raised when required columns are missing
        """
        nodes_col = {'id'}
        units_col = {'id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min'}
        lines_col = {'id', 'from', 'to', 'x', 'floor', 'cap'}
        loads_col = {'id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'}

        self.assert_cols(nodes_col, getattr(self, "nodes"))
        self.assert_cols(units_col, getattr(self, "units"))
        self.assert_cols(lines_col, getattr(self, "lines"))
        self.assert_cols(loads_col, getattr(self, "loads"))

    @staticmethod
    def assert_cols(min_col_set: set, df_to_check: pd.DataFrame) -> None:
        """Assert that a DataFrame contains at least a set of columns
        
        Args:
            min_col_set: minimal required column set
            df_to_check: DataFrame to check
            
        Raises:
            AssertionError: raised when required columns are missing
        """
        assert set(df_to_check.columns) & min_col_set == min_col_set, \
            f'Minimum required columns: {min_col_set}'

    def get_nodes_id(self, *args) -> List[np.ndarray]:
        """Get node ID mapping
        
        Args:
            *args: node ID arrays
            
        Returns:
            list of mapped node IDs
        """
        return [self.nodes.loc[np.array(a).flatten(), '#id'].values for a in args]

    # =============================================================================
    # Matrix computation methods
    # =============================================================================

    def get_node_gsdf(self) -> pd.DataFrame:
        """Get node GSDF matrix (cached)
        
        Returns:
            generation shift distribution factor matrix per node
        """
        if self._node_gsdf is None:
            self.cal_gsdf(False)
            self._node_gsdf = self.node_gsdf
        return self._node_gsdf

    def get_node_ptdf(self) -> pd.DataFrame:
        """Get node PTDF matrix (cached)
        
        Returns:
            power transfer distribution factor matrix per node
        """
        if self._node_ptdf is None:
            self.cal_gsdf(True)
            self._node_ptdf = self.node_ptdf
        return self._node_ptdf

    def cal_gsdf(self, ptdf: bool = False) -> 'ClearCase':
        """Compute GSDF and optionally PTDF matrices
        
        Args:
            ptdf: whether to compute PTDF matrices as well
        
        Returns:
            self to support method chaining
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()

        # Build line-node incidence matrix
        lines_nodes = coo_matrix(
            (np.repeat((1, -1), len(self.lines)),
             (np.tile(np.arange(len(self.lines)), 2),
              self.lines[['#from', '#to']].values.T.flatten()))
        ).toarray()

        # Line susceptance matrix
        lines_B = np.diag(-1 / self.lines['x'])

        # Node admittance matrix (excluding reference node)
        self.node_Y = lines_nodes.T.dot(lines_B).dot(lines_nodes)[1:, 1:]
        node_Y_inv = np.linalg.inv(self.node_Y)

        # Compute GSDF matrix
        node_gsdf_1 = lines_B.dot(lines_nodes.dot(
            np.vstack((np.zeros(len(self.nodes) - 1), node_Y_inv))
        ))
        self.node_gsdf = pd.DataFrame(
            np.hstack((np.zeros((len(self.lines), 1)), node_gsdf_1)),
            index=self.lines['id'],
            columns=self.nodes['id']
        )

        # GSDF matrices for units and loads
        self.unit_gsdf = self.node_gsdf[self.units['bus_id']]
        self.unit_load_gsdf = pd.concat((
            self.node_gsdf[self.units['bus_id']],
            -self.node_gsdf[self.loads['bus_id']]
        ), axis=1)

        # Compute PTDF matrix (only for constrained lines)
        if ptdf:
            self.ptdf_lines_flag = np.where(
                (self.lines.loc[:, 'floor'] == 0) & (self.lines.loc[:, 'cap'] == 0),
                False, True
            )
            self.ptdf_lines = self.lines.loc[self.ptdf_lines_flag]
            self.node_ptdf = self.node_gsdf[self.ptdf_lines_flag]
            self.unit_load_ptdf = pd.concat((
                self.node_ptdf[self.units['bus_id']],
                -self.node_ptdf[self.loads['bus_id']]
            ), axis=1)
        return self

    def get_A_matrix(self) -> np.ndarray:
        """Get adjacency matrix
        
        Returns:
            network adjacency matrix
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()
        row = np.hstack((self.lines['#from'], self.lines['#to']))
        col = np.hstack((self.lines['#to'], self.lines['#from']))
        A = coo_matrix(
            (np.ones(len(row)), (row, col)),
            shape=(len(self.nodes), len(self.nodes))
        ).toarray()
        return A

    def get_D_matrix(self) -> np.ndarray:
        """Get degree matrix
        
        Returns:
            diagonal degree matrix whose diagonal is node degrees
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()
        A = self.get_A_matrix()
        d = np.sum(A, axis=1)
        D = np.diag(d)
        return D

    def normalize_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix
        
        Args:
            adjacency: adjacency matrix
            
        Returns:
            normalized adjacency matrix
        """
        adj = sp.coo_matrix(adjacency)
        rowsum = np.array(adj.sum(1))  # node degree per row
        rowsum[rowsum == 0] = 1  # avoid division by zero
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        normalized_adjacency = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return normalized_adjacency.todense()

    def get_L_matrix(self) -> np.ndarray:
        """Get normalized Laplacian matrix
        
        Returns:
            normalized Laplacian matrix
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()
        A = self.get_A_matrix()
        A_self = A + np.eye(len(A))
        D_half_inv = np.diag(np.power(np.sum(A_self, axis=1), -0.5))
        L = np.eye(A_self.shape[0]) - np.dot(np.dot(D_half_inv, A_self), D_half_inv)
        return L

    # =============================================================================
    # Node mapping methods
    # =============================================================================

    def get_nodes_map_units_and_loads(self) -> np.ndarray:
        """Get node-to-units and node-to-loads mapping matrix
        
        Returns:
            mapping matrix (units positive, loads negative)
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()

        # Node-to-unit relationship
        self.nodes_map_units = coo_matrix(
            (np.ones(len(self.units)), (self.units['#bus_id'], self.units['#id'])),
            shape=(len(self.nodes), len(self.units))
        ).toarray()

        # Node-to-load relationship
        self.nodes_map_loads = coo_matrix(
            (np.ones(len(self.loads)), (self.loads['#bus_id'], self.loads['#id'])),
            shape=(len(self.nodes), len(self.loads))
        ).toarray()

        return np.hstack([self.nodes_map_units, -self.nodes_map_loads])

    def get_nodes_units_map(self) -> np.ndarray:
        """Get node-to-units mapping matrix
        
        Returns:
            node-to-units mapping matrix
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()

        self.nodes_map_units = coo_matrix(
            (np.ones(len(self.units)), (self.units['#bus_id'], self.units['#id'])),
            shape=(len(self.nodes), len(self.units))
        ).toarray()
        return self.nodes_map_units

    def get_nodes_loads_map(self) -> np.ndarray:
        """Get node-to-loads mapping matrix
        
        Returns:
            node-to-loads mapping matrix
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()

        self.nodes_map_loads = coo_matrix(
            (np.ones(len(self.loads)), (self.loads['#bus_id'], self.loads['#id'])),
            shape=(len(self.nodes), len(self.loads))
        ).toarray()
        return self.nodes_map_loads

    def get_nodes_flexloads_map(self) -> np.ndarray:
        """Get node-to-flexloads mapping matrix
        
        Returns:
            node-to-flexloads mapping matrix
        """
        if not hasattr(self, 'init_flag') or not self.init_flag:
            self.init()

        self.nodes_map_flexloads = coo_matrix(
            (np.ones(len(self.flexloads)), (self.flexloads['#bus_id'], self.flexloads['#id'])),
            shape=(len(self.nodes), len(self.flexloads))
        ).toarray()
        return self.nodes_map_flexloads

    # =============================================================================
    # Utility methods
    # =============================================================================

    def length(self) -> List[int]:
        """Get lengths of each data table
        
        Returns:
            [nodes, units, lines, loads]
        """
        return [len(getattr(self, v)) for v in ["nodes", "units", "lines", "loads"]]

    def to_str(self) -> List[str]:
        """Convert to string representation
        
        Returns:
            list of string representations of tables
        """
        v_list = [getattr(self, v).set_label(v) for v in ["nodes", "units", "lines", "loads"]]
        v_str_list = [str(v) for v in v_list]
        return v_str_list

    # =============================================================================
    # Cost calculation methods
    # =============================================================================
    def cal_unit_piecewise_by_margin_cost(self, band_num: int = 5) -> np.ndarray:
        """Compute piecewise linear parameters based on marginal cost
        
        Args:
            band_num: number of segments
            
        Returns:
            piecewise cost matrix
        """
        a, b, c = self.units['mc_a'], self.units['mc_b'], self.units['mc_c']
        p_max, p_min = self.units['p_max'], self.units['p_min']

        # Compute segment ratios
        xs = np.tile((p_max - p_min) / p_max / band_num, (band_num, 1)).T
        ys = []

        # Compute marginal cost per segment
        for x in np.cumsum(xs, 1).T:
            y = list(a * (p_min + p_max * x) ** 2 + b * (p_min + p_max * x) + c)
            # Ensure costs are monotonically increasing
            if len(ys) > 0:
                for i in range(len(y)):
                    if y[i] <= ys[-1][i]:
                        y[i] = ys[-1][i] + 1
            ys.append(y)

        return np.array(ys).T

    def cal_units_cost(self, units_power: Union[List, np.ndarray]) -> np.ndarray:
        """Compute generation cost for units
        
        Args:
            units_power: unit output array
            
        Returns:
            unit generation cost array
        """
        units_power = np.array(units_power) if isinstance(units_power, list) else units_power
        units_cost = (
                (self.units['mc_a'].values / 3) * units_power ** 3 +
                (self.units['mc_b'].values / 2) * units_power ** 2 +
                self.units['mc_c'].values * units_power
        )
        return units_cost

    # =============================================================================
    # Graph connectivity methods
    # =============================================================================
    
    def is_connected_graph(self) -> bool:
        """Check if the power network forms a connected graph.

        Returns:
            True if the graph is connected, False otherwise
        """
        return len(self.get_graph_components()) <= 1

    def get_graph_components(self) -> List[List]:
        """Get all connected components of the power network graph.

        Only in-service lines (status == 1, or missing status column) are
        considered.

        Returns:
            List of connected components, each component is a list of node IDs
        """
        if len(self.nodes) == 0:
            return []

        adjacency: dict = {node_id: [] for node_id in self.nodes['id'].values}

        for _, line in self.lines.iterrows():
            if line.get('status', 1) == 1:
                from_bus, to_bus = line['from'], line['to']
                adjacency[from_bus].append(to_bus)
                adjacency[to_bus].append(from_bus)

        visited: set = set()
        components: List[List] = []

        for node_id in self.nodes['id'].values:
            if node_id not in visited:
                component = []
                queue = [node_id]
                visited.add(node_id)
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)

        return components

    def get_3ph_matrix(self, config: pd.Series, matrix_type: str = 'Z') -> np.ndarray:
        """Extract 3x3 matrix from line configuration
        
        For three-phase systems, extract impedance (Z), capacitance (C), or other 
        3x3 matrices from line configuration data.
        
        Args:
            config: Line configuration series with elements like Z11, Z12, ..., Z33
            matrix_type: Type of matrix to extract ('Z' for impedance, 'C' for capacitance, etc.)
            
        Returns:
            3x3 numpy array (complex128 for 'Z', float64 for others)
            
        Raises:
            AttributeError: If line_config does not exist for this case
            KeyError: If required matrix elements are not in config
            
        Example:
            >>> config = case.line_config.iloc[0]
            >>> z_matrix = case.get_3ph_matrix(config, 'Z')
            >>> z_matrix.shape
            (3, 3)
        """
        if not hasattr(self, 'line_config'):
            raise AttributeError(
                f"Case {self.name} does not have 'line_config' attribute. "
                "This method is only available for three-phase cases."
            )
        
        # Create matrix with appropriate data type
        dtype = np.complex128 if matrix_type == 'Z' else np.float64
        
        # Generate all element names: Z11, Z12, ..., Z33
        element_names = [f'{matrix_type}{i + 1}{j + 1}' for i in range(3) for j in range(3)]
        
        # Validate all elements exist (fail fast)
        missing_elements = [name for name in element_names if name not in config.index]
        if missing_elements:
            raise KeyError(
                f"Elements {missing_elements} not found in config. "
                f"Available elements: {list(config.index)}"
            )
        
        # Extract values and reshape to 3x3 matrix (vectorized)
        matrix_3x3 = np.array([config[name] for name in element_names], dtype=dtype).reshape(3, 3)
        
        return matrix_3x3
    
    def get_line_impedance_matrices(self, reshape_2d: bool = False) -> np.ndarray:
        """Get impedance matrices for all lines in three-phase system
        
        For each line, extract its 3x3 impedance matrix from line_config and 
        multiply by line length to get total impedance.
        
        Args:
            reshape_2d: If True, reshape output to (n_lines * 3, 3) instead of (n_lines, 3, 3)
            
        Returns:
            numpy array of shape (n_lines, 3, 3) or (n_lines * 3, 3) if reshape_2d=True
            Each element is the total impedance matrix (config impedance * length)
            
        Raises:
            AttributeError: If line_config or lines does not exist
            
        Example:
            >>> z_matrices = case.get_line_impedance_matrices()
            >>> z_matrices.shape
            (113, 3, 3)  # For 113 lines
            >>> z_matrices_2d = case.get_line_impedance_matrices(reshape_2d=True)
            >>> z_matrices_2d.shape
            (339, 3)  # 113 * 3 rows, 3 columns
        """
        if not hasattr(self, 'line_config'):
            raise AttributeError(
                f"Case {self.name} does not have 'line_config' attribute. "
                "This method is only available for three-phase cases."
            )
        
        if not hasattr(self, 'lines'):
            raise AttributeError(
                f"Case {self.name} does not have 'lines' attribute."
            )
        
        # Build impedance matrices using list comprehension
        z_matrices = np.array([
            self.get_3ph_matrix(self.line_config.loc[int(line['config_name'])]) * line['length']
            for _, line in self.lines.iterrows()
        ])
        
        # Optionally reshape to 2D
        if reshape_2d:
            return z_matrices.reshape(-1, 3)
        
        return z_matrices

