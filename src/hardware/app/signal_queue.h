#ifndef QUEUE_H
#define QUEUE_H

#include <cstdint>
#include <list>


class SignalQueue {
    public:
        SignalQueue(uint32_t maxLength, uint32_t channels);
        void add(uint16_t values[]);
        void notifyOnOverflowingElement(
            uint32_t atEveryNthElement,
            void (*eventHandler)(SignalQueue*));
        void copyToBuffer(float buffer[]);
    private:
        const uint32_t maxLength, channels;
        std::list<uint16_t> rawQueue;
        uint32_t overflowCounter, notifyAtEveryNthOverflow;
        void (*eventHandler)(SignalQueue*);
        bool isFilled();
        void removeHead();
        bool hasEventHandler();
        bool overflowNotificationRequired();
};

#endif