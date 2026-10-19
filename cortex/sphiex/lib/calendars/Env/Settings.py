# -*- coding: utf-8 -*-

from ..DateUtilities import Date


class SettingsFactory(object):

    def __init__(self):
        self._evaluationDate = None
        self._includeToday = None

    @property
    def evaluationDate(self):
        if not self._evaluationDate:
            return Date.todaysDate()
        return self._evaluationDate

    @evaluationDate.setter
    def evaluationDate(self, value):
        #require(isinstance(value, Date), ValueError, "{0} is not a valid PyFin date object".format(value))
        self._evaluationDate = value

    @property
    def includeTodaysCashFlows(self):
        return self._includeToday

    @includeTodaysCashFlows.setter
    def includeTodaysCashFlows(self, value):
        self._evaluationDate = value

    def resetEvaluationDate(self):
        self._evaluationDate = None

    def anchorEvaluationDate(self):
        if self._evaluationDate is None:
            self._evaluationDate = Date.todaysDate()


Settings = SettingsFactory()
