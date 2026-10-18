#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <iostream>
#include <codecvt>
#include <condition_variable>
#include <locale>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <vector>
#include <stdexcept>

#if !defined(__APPLE__) && !defined(_MSC_VER)
#include <iconv.h>
#endif

#ifdef __APPLE__
#include <iconv.h>
#endif

#include "pybind11/pybind11.h"


using namespace std;
using namespace pybind11;


//任务结构体
struct Task
{
    int task_name = 0;		//回调函数名称对应的常数
    void *task_data = nullptr;	//任务指针
    void *task_error = nullptr;	//错误指针
    int task_id = 0;		//任务id
    bool task_last = false;		//是否为最后返回
};

class TerminatedError : std::exception
{};

class TaskQueue
{
private:
    queue<Task> queue_;						//标准队列
    mutex mutex_;							//锁
    condition_variable cond_;				//条件变量

    bool _terminate = false;

public:

    //存入新的任务
    void push(const Task &task)
    {
        unique_lock<mutex > mlock(mutex_);
        queue_.push(task);					//存入任务后写入队列
        mlock.unlock();						//释放锁
        cond_.notify_one();					//通知正在等待的线程
    }

    //取出最早的任务
    Task pop()
    {
        unique_lock<mutex> mlock(mutex_);
        cond_.wait(mlock, [&]() {
            return !queue_.empty() || _terminate;
        });				//等待条件变量通知
        if (_terminate)
            throw TerminatedError();
        Task task = queue_.front();			//获取队列中最早的一个任务
        queue_.pop();						//删除任务
        return task;						//返回该任务
    }

    void terminate()
    {
        {
            lock_guard<mutex> lock(mutex_);
            _terminate = true;
        }
        cond_.notify_all();					//通知正在等待的线程
    }
};


//从字典中获取某个键值对应的整数，并将其赋值到请求结构体对象上
void getInt(const dict &d, const char *key, int *value)
{
    if (d.contains(key))		//检查键值是否存在该键值
    {
        object o = d[key];		//获取该键值
        *value = o.cast<int>();
    }
};


//从字典中获取某个键值对应的浮点数，并将其赋值到请求结构体对象上
void getDouble(const dict &d, const char *key, double *value)
{
    if (d.contains(key))
    {
        object o = d[key];
        *value = o.cast<double>();
    }
};


//从字典中获取某个键值对应的字符，并将其赋值到请求结构体对象上
void getChar(const dict &d, const char *key, char *value)
{
    if (d.contains(key))
    {
        object o = d[key];
        *value = o.cast<char>();
    }
};


template <size_t size>
using string_literal = char[size];

//从字典中获取某个键值对应的字符串，并将其赋值到请求结构体对象上
template <size_t size>
void getString(const pybind11::dict &d, const char *key, string_literal<size> &value)
{
    if (d.contains(key))
    {
        object o = d[key];
        string s = o.cast<string>();
        const size_t count = min(s.size(), size - 1);
        memcpy(value, s.data(), count);
        value[count] = '\0';
    }
};


//将GBK编码的字符串转换为UTF8
#ifndef __APPLE__
inline string toUtf(const string &gb2312)
{

    #ifdef _MSC_VER
        const static locale loc("zh-CN");
        vector<wchar_t> wstr(gb2312.size());
        wchar_t* wstrEnd = nullptr;
        const char* gbEnd = nullptr;
        mbstate_t state = {};
        int res = use_facet<codecvt<wchar_t, char, mbstate_t> >
            (loc).in(state,
                gb2312.data(), gb2312.data() + gb2312.size(), gbEnd,
                wstr.data(), wstr.data() + wstr.size(), wstrEnd);

        if (codecvt_base::ok == res)
        {
            wstring_convert<codecvt_utf8<wchar_t>> cutf8;
            return cutf8.to_bytes(wstring(wstr.data(), wstrEnd));
        }
        return string();
    #else
        if (all_of(gb2312.begin(), gb2312.end(), [](unsigned char c) { return c < 0x80; }))
        {
            return gb2312;
        }

        iconv_t converter = iconv_open("UTF-8", "GB18030");
        if (converter == (iconv_t)-1)
        {
            return string();
        }
        size_t input_left = gb2312.size();
        size_t output_left = gb2312.size() * 4 + 1;
        string output(output_left, '\0');
        char *input = const_cast<char *>(gb2312.data());
        char *output_ptr = output.data();
        size_t result = iconv(converter, &input, &input_left, &output_ptr, &output_left);
        iconv_close(converter);
        if (result == (size_t)-1)
        {
            return string();
        }
        output.resize(static_cast<size_t>(output_ptr - output.data()));
        return output;
    #endif
}
#else
iconv_t cd = iconv_open("UTF-8", "GB2312");

int code_convert(char *inbuf, size_t inlen, char *outbuf, size_t outlen) 
{
    char **pin = &inbuf;
    char **pout = &outbuf;

    memset(outbuf, 0, outlen);

    if ((int)iconv(cd, pin, &inlen, pout, &outlen) == -1)
    {
        return -1;
    }
    *pout = "\0";

    return 0;
}

inline string toUtf(const string &gb2312)
{
    int length = gb2312.size() * 2 + 1;
    char temp[length];

    int n = code_convert((char*)gb2312.c_str(), gb2312.size(), temp, length);

    if(n == 0)
    {
        return temp;
    }
    else
    {
        return "";
    }
}
#endif
