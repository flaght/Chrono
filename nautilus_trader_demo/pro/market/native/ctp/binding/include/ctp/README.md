# CTP headers

These three headers are the market-data subset required to compile
`bomber_ctp_md`:

- `ThostFtdcMdApi.h`
- `ThostFtdcUserApiStruct.h`
- `ThostFtdcUserApiDataType.h`

They were copied from the same CTP 6.7.11 SDK bundle already present under
`pro/3rd/vnpy_ctp`. Keep the headers and `libthostmduserapi_se.so` on the same
SDK version when upgrading.
