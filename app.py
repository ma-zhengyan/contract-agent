import streamlit as st
from core_agent import extract_text_from_file, run_agent

st.set_page_config(page_title="合同风险智能审核", page_icon="📄", layout="wide")
st.title("📄 合同风险智能审核系统")
st.caption("基于 LangGraph · 支持 PDF / Word / Excel / PPT / TXT")

with st.sidebar:
    st.header("🔑 配置")
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        api_key = None
        st.error("未检测到 DeepSeek API Key，请联系管理员")
    st.markdown("---")
    st.markdown("**支持格式**：PDF, DOCX, XLSX, PPTX, TXT")
    st.markdown("**审核维度**：违约责任、赔偿上限、保密义务")

st.subheader("📤 上传文件")
uploaded = st.file_uploader("点击或拖拽", type=["pdf", "docx", "xlsx", "pptx", "txt"], label_visibility="collapsed")

# 初始化 session_state
if "contract_input" not in st.session_state:
    st.session_state.contract_input = ""

contract_text = ""

if uploaded:
    with st.spinner("解析中..."):
        text = extract_text_from_file(uploaded)
    if text:
        contract_text = text
        st.session_state.contract_input = text  # 同步到 session
        st.success(f"✅ 提取成功，共 {len(text)} 字符")
        with st.expander("预览文本"):
            st.text(text[:1500] + ("..." if len(text) > 1500 else ""))
    else:
        st.warning("解析为空，请手动粘贴")

# 手动输入框（绑定 session_state）
contract_text = st.text_area(
    "📝 或手动粘贴合同",
    value=st.session_state.contract_input,
    height=200,
    placeholder="粘贴合同内容..."
)

# 载入示例按钮（直接写入 session_state）
if st.button("📌 载入示例"):
    st.session_state.contract_input = """采购框架协议
甲方：XX科技  乙方：YY供应链
第一条 违约责任：逾期每日千分之一，总额不超5%
第二条 赔偿上限：不超过总额10%，仅赔直接损失
第三条 保密义务：期限为终止后5年"""
    st.rerun()

# 如果 session_state 有内容，确保输入框显示
if st.session_state.contract_input:
    contract_text = st.session_state.contract_input

    
# ========== 审核按钮 ==========
if st.button("🚀 开始审核", type="primary"):
    if not api_key:
        st.error("请先输入API Key")
    elif not contract_text.strip():
        st.error("请提供合同内容")
    else:
        with st.status("审核中...", expanded=True) as status:
            try:
                # ----- 执行审核，result 在此定义 -----
                result = run_agent(contract_text)
                status.update(label="✅ 完成", state="complete")
            except Exception as e:
                st.error(f"❌ 审核失败: {e}")
                st.stop()

        # ----- 报告展示（此时 result 肯定存在）-----
        st.subheader("📊 报告")

        risks_raw = result.get("risk_report", [])

        # 处理各种数据类型
        if isinstance(risks_raw, list):
            risks = risks_raw
        elif isinstance(risks_raw, dict):
            if "risks" in risks_raw and isinstance(risks_raw["risks"], list):
                risks = risks_raw["risks"]
            elif "risk" in risks_raw:
                risks = [risks_raw]
            else:
                st.warning("返回的字典中未找到风险列表，显示完整内容")
                st.json(result)
                st.stop()
        else:
            st.info("未发现风险或数据格式不支持")
            st.stop()

        # 过滤非字典元素
        risks = [r for r in risks if isinstance(r, dict)]

        if not risks:
            st.info("未发现风险")
        else:
            # 兼容中英文键名
            def get_level(item):
                return item.get("level") or item.get("风险等级") or "未知"
            def get_risk(item):
                return item.get("risk") or item.get("风险分析") or item.get("风险描述") or "未知风险"
            def get_suggestion(item):
                return item.get("suggestion") or item.get("建议") or "无建议"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总数", len(risks))
            c2.metric("🔴 高", sum(1 for r in risks if get_level(r) == "高"))
            c3.metric("🟡 中", sum(1 for r in risks if get_level(r) == "中"))
            c4.metric("🟢 低", sum(1 for r in risks if get_level(r) == "低"))

            for i, item in enumerate(risks, 1):
                level = get_level(item)
                risk_text = get_risk(item)
                suggestion = get_suggestion(item)
                icon = "🔴" if level == "高" else "🟡" if level == "中" else "🟢"
                with st.expander(f"{icon} 风险{i}: {risk_text[:30]}... ({level})", expanded=(level == "高")):
                    st.write(f"**风险描述**：{risk_text}")
                    st.write(f"**修改建议**：{suggestion}")

            high_count = sum(1 for r in risks if get_level(r) == "高")
            if high_count > 0:
                st.warning(f"⚠️ 检测到 {high_count} 个高风险条款，已触发人机协作（HITL）机制，建议人工复核。")
            else:
                st.success("✅ 未发现高风险条款，系统已完成自动审核。")

            with st.expander("🔍 完整JSON"):
                st.json(result)