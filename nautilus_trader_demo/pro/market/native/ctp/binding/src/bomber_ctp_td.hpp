#pragma once

#include <pybind11/pybind11.h>
#include "ctp/ThostFtdcTraderApi.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// Narrow TraderApi surface used by CtpTdApiTransport. Native callbacks are copied
// into a queue and delivered to the Python subclass from one dispatcher thread.
class TdApi final : public CThostFtdcTraderSpi {
public:
    TdApi() = default;
    TdApi(const TdApi &) = delete;
    TdApi &operator=(const TdApi &) = delete;
    ~TdApi();

    void createFtdcTraderApi(const std::string &path = "", bool production = true);
    void registerFront(const std::string &front);
    void subscribePrivateTopic(int mode);
    void subscribePublicTopic(int mode);
    void init();
    void release();
    int exit();

    int reqAuthenticate(const pybind11::dict &fields, int id);
    int reqUserLogin(const pybind11::dict &fields, int id);
    int reqSettlementInfoConfirm(const pybind11::dict &fields, int id);
    int reqQryInvestorPosition(const pybind11::dict &fields, int id);
    int reqQryTradingAccount(const pybind11::dict &fields, int id);
    int reqQryOrder(const pybind11::dict &fields, int id);
    int reqOrderInsert(const pybind11::dict &fields, int id);
    int reqOrderAction(const pybind11::dict &fields, int id);

    void OnFrontConnected() override;
    void OnFrontDisconnected(int reason) override;
    void OnRspError(CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspAuthenticate(CThostFtdcRspAuthenticateField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspUserLogin(CThostFtdcRspUserLoginField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspSettlementInfoConfirm(CThostFtdcSettlementInfoConfirmField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspQryInvestorPosition(CThostFtdcInvestorPositionField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspQryTradingAccount(CThostFtdcTradingAccountField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspQryOrder(CThostFtdcOrderField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspOrderInsert(CThostFtdcInputOrderField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRspOrderAction(CThostFtdcInputOrderActionField *data, CThostFtdcRspInfoField *error, int id, bool last) override;
    void OnRtnOrder(CThostFtdcOrderField *data) override;
    void OnRtnTrade(CThostFtdcTradeField *data) override;
    void OnErrRtnOrderInsert(CThostFtdcInputOrderField *data, CThostFtdcRspInfoField *error) override;
    void OnErrRtnOrderAction(CThostFtdcOrderActionField *data, CThostFtdcRspInfoField *error) override;

private:
    template <typename T> static std::shared_ptr<T> copy(const T *value);
    template <typename... Args> void emit(const char *name, Args&&... args);
    static THOST_TE_RESUME_TYPE resume_type(int mode);
    CThostFtdcTraderApi &required();
    void enqueue(std::function<void()> action);
    void dispatch();
    void shutdown();

    CThostFtdcTraderApi *api_ = nullptr;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    std::thread worker_;
    bool stopping_ = false;
    bool overflowed_ = false;
};
