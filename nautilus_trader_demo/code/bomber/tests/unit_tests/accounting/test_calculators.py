# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import datetime
from decimal import Decimal

import pandas as pd
import pytest

from bomber import TEST_DATA_DIR
from bomber.accounting.calculators import RolloverInterestCalculator
from bomber.core import bomber_pyo3
from bomber.model.currencies import AUD
from bomber.model.currencies import BTC
from bomber.model.currencies import JPY
from bomber.model.currencies import USD
from bomber.test_kit.stubs.data import UNIX_EPOCH
from bomber.test_kit.stubs.identifiers import TestIdStubs


AUDUSD_SIM = TestIdStubs.audusd_id()
GBPUSD_SIM = TestIdStubs.gbpusd_id()
USDJPY_SIM = TestIdStubs.usdjpy_id()


class TestExchangeRateCalculator:
    def test_get_rate_when_from_currency_equals_to_currency_returns_one(self):
        # Arrange
        bid_rates = {"AUD/USD": 0.80000}
        ask_rates = {"AUD/USD": 0.80010}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            USD.code,
            USD.code,
            bomber_pyo3.PriceType.BID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result == 1

    def test_get_rate_when_no_currency_rate_returns_zero(self):
        # Arrange
        bid_rates = {"AUD/USD": 0.80000}
        ask_rates = {"AUD/USD": 0.80010}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            USD.code,
            JPY.code,
            bomber_pyo3.PriceType.BID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result is None

    def test_get_rate(self):
        # Arrange
        bid_rates = {"AUD/USD": 0.80000}
        ask_rates = {"AUD/USD": 0.80010}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            AUD.code,
            USD.code,
            bomber_pyo3.PriceType.BID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result == pytest.approx(Decimal("0.80000"), abs=Decimal("1e-9"))

    def test_get_rate_when_symbol_has_slash(self):
        # Arrange
        bid_rates = {"AUD/USD": 0.80000}
        ask_rates = {"AUD/USD": 0.80010}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            AUD.code,
            USD.code,
            bomber_pyo3.PriceType.BID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result == pytest.approx(Decimal("0.80000"), abs=Decimal("1e-9"))

    def test_get_rate_for_inverse1(self):
        # Arrange
        bid_rates = {"BTC/USD": 10501.5}
        ask_rates = {"BTC/USD": 10500.0}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            USD.code,
            BTC.code,
            bomber_pyo3.PriceType.BID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result == pytest.approx(Decimal("9.522449173927534e-05"), abs=Decimal("1e-9"))

    def test_get_rate_for_inverse2(self):
        # Arrange
        bid_rates = {"USD/JPY": 110.100}
        ask_rates = {"USD/JPY": 110.130}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            JPY.code,
            USD.code,
            bomber_pyo3.PriceType.BID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result == pytest.approx(Decimal("0.009082652134423252"), abs=Decimal("1e-9"))

    def test_calculate_exchange_rate_by_inference(self):
        # Arrange
        bid_rates = {
            "USD/JPY": 110.100,
            "AUD/USD": 0.80000,
        }
        ask_rates = {
            "USD/JPY": 110.130,
            "AUD/USD": 0.80010,
        }

        # Act
        result1 = bomber_pyo3.get_exchange_rate(
            JPY.code,
            AUD.code,
            bomber_pyo3.PriceType.BID,
            bid_rates,
            ask_rates,
        )

        result2 = bomber_pyo3.get_exchange_rate(
            AUD.code,
            JPY.code,
            bomber_pyo3.PriceType.ASK,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result1 == pytest.approx(Decimal("0.011353315168029066"), abs=Decimal("1e-9"))
        assert result2 == pytest.approx(Decimal("88.115013"), abs=Decimal("1e-9"))

    def test_calculate_exchange_rate_for_mid_price_type(self):
        # Arrange
        bid_rates = {"USD/JPY": 110.100}
        ask_rates = {"USD/JPY": 110.130}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            JPY.code,
            USD.code,
            bomber_pyo3.PriceType.MID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result == pytest.approx(Decimal("0.009081414884438995"), abs=Decimal("1e-9"))

    def test_calculate_exchange_rate_for_mid_price_type2(self):
        # Arrange
        bid_rates = {"USD/JPY": 110.100}
        ask_rates = {"USD/JPY": 110.130}

        # Act
        result = bomber_pyo3.get_exchange_rate(
            USD.code,
            JPY.code,
            bomber_pyo3.PriceType.MID,
            bid_rates,
            ask_rates,
        )

        # Assert
        assert result == pytest.approx(Decimal("110.115"), abs=Decimal("1e-9"))


class TestRolloverInterestCalculator:
    def setup(self):
        # Fixture Setup
        self.data = pd.read_csv(TEST_DATA_DIR / "short-term-interest.csv")

    def test_rate_dataframe_returns_correct_dataframe(self):
        # Arrange
        calculator = RolloverInterestCalculator(data=self.data)

        # Act
        rate_data = calculator.get_rate_data()

        # Assert
        assert isinstance(rate_data, dict)

    def test_calc_overnight_fx_rate_with_audusd_on_unix_epoch_returns_correct_rate(
        self,
    ):
        # Arrange
        calculator = RolloverInterestCalculator(data=self.data)

        # Act
        rate = calculator.calc_overnight_rate(AUDUSD_SIM, UNIX_EPOCH)

        # Assert
        assert rate == -8.52054794520548e-05

    def test_calc_overnight_fx_rate_with_audusd_on_later_date_returns_correct_rate(
        self,
    ):
        # Arrange
        calculator = RolloverInterestCalculator(data=self.data)

        # Act
        rate = calculator.calc_overnight_rate(AUDUSD_SIM, datetime.date(2018, 2, 1))

        # Assert
        assert rate == -2.739726027397263e-07

    def test_calc_overnight_fx_rate_with_audusd_on_impossible_dates_returns_zero(self):
        # Arrange
        calculator = RolloverInterestCalculator(data=self.data)

        # Act, Assert
        with pytest.raises(RuntimeError):
            calculator.calc_overnight_rate(AUDUSD_SIM, datetime.date(1900, 1, 1))

        with pytest.raises(RuntimeError):
            calculator.calc_overnight_rate(AUDUSD_SIM, datetime.date(3000, 1, 1))
