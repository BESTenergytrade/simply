from simply.market_wrapper import (PayAsBidMatchingAlgorithm)
from simply.market import MARKET_MAKER_THRESHOLD

FLOATING_POINT_TOLERANCE = 0.00001


class TestPayAsBidMatchingAlgorithm:
    """Test the pay-as-bid matching algorithm"""

    @staticmethod
    def test_perform_simple_pay_as_bid_match():
        """
        Test whether the matches from a list of offers and bids are the expected ones.
        Single bid and single offer with floating point tolerance
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    "bids": [
                        {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20}
                    ],
                    "offers": [
                        {"id": 4, "seller": "A", "energy_rate": 1 + FLOATING_POINT_TOLERANCE,
                         "energy": 25}
                    ],
                }
            }
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1.00001, "energy": 25},
             "selected_energy": 20, "trade_rate": 3, "matching_requirements": None},
        ]
        assert recommendations == expected_recommendations

    @staticmethod
    def test_market_maker_match():
        """
        Test whether the matches from a list of offers and bids are the expected ones.
        Single bid and single offer with floating point tolerance
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    "bids": [
                        {"id": 3, "buyer": "C", "energy_rate": 3, "energy": MARKET_MAKER_THRESHOLD}
                    ],
                    "offers": [
                        {"id": 4, "seller": "A", "energy_rate": 1 + FLOATING_POINT_TOLERANCE,
                         "energy": 25}
                    ],
                }
            }
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 3, "buyer": "C", "energy_rate": 3, "energy": MARKET_MAKER_THRESHOLD},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1.00001, "energy": 25},
             "selected_energy": 25.0, "trade_rate": 3, "matching_requirements": None},
        ]
        assert recommendations == expected_recommendations

    @staticmethod
    def test_buyer_seller_same_no_pay_as_bid_match():
        """
        Test whether the matches from a list of offers and bids are the expected ones.
        If seller and buyer have the same ID they should not be matched
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    "bids": [
                        {"id": 3, "buyer": "A", "energy_rate": 3, "energy": 20}
                    ],
                    "offers": [
                        {"id": 4, "seller": "A", "energy_rate": 1, "energy": 25}
                    ],
                }
            }
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        expected_recommendations = []
        assert recommendations == expected_recommendations

    @staticmethod
    def test_perform_pay_as_bid_match_multiple_offers_bids_price_ordered():
        """
        Test whether the matches from a list of offers and bids are the expected ones.
        The orders are matched in the order of decreasing and increasing energy_rate
        for bids and asks respectively.
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    # The bid matching order should be 3, 2, 1 (decreasing energy rate) and the
                    # order matching offer should be 4, 6, 5 (increasing energy rate).
                    "bids": [
                        {"id": 1, "buyer": "A", "energy_rate": 1, "energy": 10},
                        {"id": 2, "buyer": "B", "energy_rate": 2, "energy": 15},
                        {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20},
                    ],
                    "offers": [
                        {"id": 4, "seller": "A", "energy_rate": 1 + FLOATING_POINT_TOLERANCE,
                         "energy": 25},
                        {"id": 5, "seller": "B", "energy_rate": 5, "energy": 30},
                        {"id": 6, "seller": "C", "energy_rate": 2.4, "energy": 35},
                    ],
                }

            }
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1.00001, "energy": 25},
             "selected_energy": 20, "trade_rate": 3, "matching_requirements": None},

            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 2, "buyer": "B", "energy_rate": 2, "energy": 15},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1.00001, "energy": 25},
             "selected_energy": 5, "trade_rate": 2, "matching_requirements": None},
        ]
        # ordering of matches within recommendation list is not important
        assert [i for i in recommendations if i not in expected_recommendations] == []

    @staticmethod
    def test_perform_pay_as_bid_match_multiple_offers_bids():
        """
        Test whether the matches from a list of offers and bids are the expected ones.
        The orders are matched in the order of decreasing and increasing energy_rate
        for bids and asks respectively.
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    # The bid matching order should be 3, 2, 1 (decreasing energy rate) and the
                    # order matching offer should be 4, 6, 5 (increasing energy rate).
                    "bids": [
                        {"id": 1, "buyer": "A", "energy_rate": 1, "energy": 10},
                        {"id": 2, "buyer": "B", "energy_rate": 2, "energy": 15},
                        {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20},
                    ],
                    "offers": [
                        {"id": 4, "seller": "A", "energy_rate": 1 + FLOATING_POINT_TOLERANCE,
                         "energy": 25},
                        {"id": 5, "seller": "B", "energy_rate": 5, "energy": 30},
                        {"id": 6, "seller": "C", "energy_rate": 2.4, "energy": 35},
                    ],
                }

            },
            "market2": {
                "2021-10-06T12:00": {
                    # The bid matching order should be 9, 8, 7 and the order matching offer should
                    # be 12, 11, 10 (arbitrary, but in practice the reverse of the original order
                    # for the case of the same energy rate).
                    # The matches 9-12, 8-11, 7-10 are skipped, because the buyer is the same as
                    # the seller in these cases
                    "bids": [
                        {"id": 7, "buyer": "A", "energy_rate": 1.5, "energy": 40},
                        {"id": 8, "buyer": "B", "energy_rate": 2, "energy": 45},
                        {"id": 9, "buyer": "C", "energy_rate": 6, "energy": 50},
                    ],
                    "offers": [
                        {"id": 10, "seller": "A", "energy_rate": 1, "energy": 55},
                        {"id": 11, "seller": "B", "energy_rate": 1, "energy": 60},
                        {"id": 12, "seller": "C", "energy_rate": 1, "energy": 65},
                    ],
                }
            }
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1.00001, "energy": 25},
             "selected_energy": 20, "trade_rate": 3, "matching_requirements": None},

            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 2, "buyer": "B", "energy_rate": 2, "energy": 15},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1.00001, "energy": 25},
             "selected_energy": 5, "trade_rate": 2, "matching_requirements": None},

            {"market_id": "market2",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 8, "buyer": "B", "energy_rate": 2, "energy": 45},
             "offer": {"id": 12, "seller": "C", "energy_rate": 1, "energy": 65},
             "selected_energy": 45, "trade_rate": 2, "matching_requirements": None},

            {"market_id": "market2",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 7, "buyer": "A", "energy_rate": 1.5, "energy": 40},
             "offer": {"id": 12, "seller": "C", "energy_rate": 1, "energy": 65},
             "selected_energy": 20, "trade_rate": 1.5, "matching_requirements": None},

            {"market_id": "market2",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 9, "buyer": "C", "energy_rate": 6, "energy": 50},
             "offer": {"id": 11, "seller": "B", "energy_rate": 1, "energy": 60},
             "selected_energy": 50, "trade_rate": 6, "matching_requirements": None},

            {"market_id": "market2",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 7, "buyer": "A", "energy_rate": 1.5, "energy": 40},
             "offer": {"id": 11, "seller": "B", "energy_rate": 1, "energy": 60},
             "selected_energy": 10, "trade_rate": 1.5, "matching_requirements": None},

        ]
        # ordering of matches within recommendation list is not important
        assert [i for i in recommendations if i not in expected_recommendations] == []

    @staticmethod
    def test_energy_unit():
        """
        Test the granularity of energy is matched.
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    "bids": [
                        {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20.001}
                    ],
                    "offers": [
                        {"id": 4, "seller": "A", "energy_rate": 1.0, "energy": 25}
                    ],
                }
            }
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20.001},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1, "energy": 25},
             "selected_energy": 20.001, "trade_rate": 3, "matching_requirements": None},
        ]
        assert recommendations == expected_recommendations

    @staticmethod
    def test_perform_pay_as_bid_match_single_offer_bid():
        """
        Test whether a single offer can match with a bid.
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    "bids": [{"id": 1, "buyer": "B", "energy_rate": 3, "energy": 20}],
                    "offers": [{"id": 2, "seller": "A", "energy_rate": 3.1, "energy": 30}],
                }
            },
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        assert not recommendations
        # Adapting the offer energy rate to a value that can match with the bid
        data["market1"]["2021-10-06T12:00"]["offers"][0]["energy_rate"] = 1
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 1, "buyer": "B", "energy_rate": 3, "energy": 20},
             "offer": {"id": 2, "seller": "A", "energy_rate": 1, "energy": 30},
             "selected_energy": 20, "trade_rate": 3, "matching_requirements": None}
        ]
        assert recommendations == expected_recommendations

    @staticmethod
    def test_perform_pay_as_bid_match_single_offer_multiple_bids():
        """
        Test whether a single offer can match with multiple bids.
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    "bids": [
                        {"id": 1, "buyer": "B", "energy_rate": 3, "energy": 20},
                        {"id": 2, "buyer": "C", "energy_rate": 3, "energy": 20},
                        {"id": 3, "buyer": "A", "energy_rate": 3, "energy": 20},
                    ],
                    "offers": [{"id": 4, "seller": "A", "energy_rate": 1, "energy": 70}],
                }
            },
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        # The bid-id 3 will not match because it has the same buyer as the seller
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 2, "buyer": "C", "energy_rate": 3, "energy": 20},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1, "energy": 70},
             "selected_energy": 20, "trade_rate": 3, "matching_requirements": None},
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 1, "buyer": "B", "energy_rate": 3, "energy": 20},
             "offer": {"id": 4, "seller": "A", "energy_rate": 1, "energy": 70},
             "selected_energy": 20, "trade_rate": 3, "matching_requirements": None},
        ]
        # ordering of matches within recommendation list is not important
        assert [i for i in recommendations if i not in expected_recommendations] == []

    @staticmethod
    def test_perform_pay_as_bid_match_single_bid_multiple_offers():
        """
        Test whether a single bid can match with multiple offers.
        """
        data = {
            "market1": {
                "2021-10-06T12:00": {
                    "bids": [{"id": 1, "buyer": "B", "energy_rate": 3, "energy": 70}],
                    "offers": [
                        {"id": 10, "seller": "A", "energy_rate": 1, "energy": 20},
                        {"id": 11, "seller": "B", "energy_rate": 1, "energy": 20},
                        {"id": 12, "seller": "C", "energy_rate": 1, "energy": 10},
                    ],
                }
            },
        }
        recommendations = PayAsBidMatchingAlgorithm.get_matches_recommendations(data)
        # The offer-id 11 will not match because it has the same seller as the bid buyer
        expected_recommendations = [
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 1, "buyer": "B", "energy_rate": 3, "energy": 70},
             "offer": {"id": 12, "seller": "C", "energy_rate": 1, "energy": 10},
             "selected_energy": 10, "trade_rate": 3, "matching_requirements": None},
            {"market_id": "market1",
             "time_slot": "2021-10-06T12:00",
             "bid": {"id": 1, "buyer": "B", "energy_rate": 3, "energy": 70},
             "offer": {"id": 10, "seller": "A", "energy_rate": 1, "energy": 20},
             "selected_energy": 20, "trade_rate": 3, "matching_requirements": None},
        ]
        # ordering of matches within recommendation list is not important
        assert [i for i in recommendations if i not in expected_recommendations] == []
