import asyncio
from workflows.prevention_flow import build_prevention_flow
from workflows.response_flow import build_response_flow


async def main():
    # 初始化工作流
    prevention_app = build_prevention_flow()
    response_app = build_response_flow()

    # 执行预警流程
    prevention_state = {
        "location": "广东省深圳市",
        "wind_speed": "40m/s",
        "messages": []
    }
    prevention_result = await prevention_app.ainvoke(prevention_state)

    # 条件触发响应流程
    if prevention_result.get("trigger_response"):
        response_state = {
            **prevention_result,
            "sensor_data": get_live_sensors_data()
        }
        await response_app.ainvoke(response_state)


if __name__ == "__main__":
    asyncio.run(main())