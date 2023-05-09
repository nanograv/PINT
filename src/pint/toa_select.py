"""Tool for selecting a subset of TOAs."""
import numpy as np

__all__ = ["TOASelect"]


class TOASelect:
    """Select toas from toa table based on a given condition.

    The selection result will be saved in the `select_result`
    attribute as a mini caching for the future calculation.

    Parameters
    ----------
    is_range: bool
        Is this toa selection a range selection.
    use_hash: bool, optional [default: False]
        If use hash for caching.

    Note
    ----
    The supported condition types are:
        - Ranged condition in the format of {'DMX_0001':(54000, 54001), ...}
        - Key condition in the format of {'JUMP1': 'L-wide', ...}

    Putting an object as condition will slow the process dramatically.

    """

    def __init__(self, is_range, use_hash=False):
        self.is_range = is_range
        self.use_hash = use_hash
        self.hash_dict = {}
        self.columns_info = {}
        self.select_result = {}

    def check_condition(self, new_cond):
        """Check if the condition that same with old input.

        The new condition's
        information will be updated to the 'condition' attribute.

        Parameters
        ----------
        new_cond : dict
            New condition for selection.

        """
        condition_chg = {}
        condition_unchg = {}
        if not hasattr(self, "condition"):
            self.condition = new_cond
            condition_chg = new_cond
        else:
            old = set(self.condition.items())
            new = set(new_cond.items())
            # Get the condition entries have not been changed
            unchg = set.intersection(old, new)
            # Get the condition entries have been changed
            chg = new - old
            condition_chg = dict(chg)
            condition_unchg = dict(unchg)
            self.condition.update(dict(chg))
        return condition_unchg, condition_chg

    def check_table_column(self, new_column):
        """check if a table column has been changed from the old one.

        The column information will be updated to the new column if they
        are not the same.

        Parameters
        ----------
        column: toas.table column
            The toa table column that the condition is applied on

        Returns
        -------
        bool
            True for column is the same as old one
            False for column has been changed.
        """
        if self.use_hash:
            if new_column.name in self.hash_dict.keys() and self.hash_dict[
                new_column.name
            ] == hash(new_column.tobytes()):
                return True
            # update hash value to new column
            self.hash_dict[new_column.name] = hash(new_column.tobytes())
        elif new_column.name not in self.columns_info.keys():
            self.columns_info[new_column.name] = new_column
        elif np.array_equal(self.columns_info[new_column.name], new_column):
            return True
        else:
            self.columns_info[new_column.name] = new_column
        return False

    def get_select_range(self, condition, column):
        """
        A function get the selected toa index via a range comparison.
        """
        result = {}
        for k, v in condition.items():
            msk = np.logical_and(column >= v[0], column <= v[1])
            result[k] = np.where(msk)[0]
        return result

    def get_select_non_range(self, condition, column):
        """
        A function get the selected toa index via compare the key value.
        """
        result = {}
        for k, v in condition.items():
            index = np.where(column == v)[0]
            result[k] = index
        return result

    def get_select_index(self, condition, column):
        # Check if condition get changed
        cd_unchg, cd_chg = self.check_condition(condition)
        if col_change := self.check_table_column(column):
            if self.is_range:
                new_select = self.get_select_range(cd_chg, column)
            else:
                new_select = self.get_select_non_range(cd_chg, column)
            self.select_result.update(new_select)
            return {k: self.select_result[k] for k in condition.keys()}

        else:
            if self.is_range:
                new_select = self.get_select_range(condition, column)
            else:
                new_select = self.get_select_non_range(condition, column)
            self.select_result = new_select
            return new_select
