import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.simulate_prompt_injection import generate_injection_audit_md


def test_generate_injection_audit_md(tmp_path: Path):
    output = tmp_path / "audit.md"
    generated = generate_injection_audit_md(output)
    assert generated.exists()
    content = generated.read_text(encoding="utf-8")
    assert "两类提示词实际注入模拟报告（无模型调用）" in content
    assert "LLM_CALLED: false" in content
    assert "## 3. 城市更新注入明细" in content
    assert "## 4. 空间分析注入明细" in content
    assert "## 6. 同名策略并排差异总表" in content
    assert "### 7.1 策略 `cot`" in content
    assert "### 7.2 策略 `few`" in content
    assert "### 7.3 策略 `one`" in content
    assert "### 7.4 策略 `zero`" in content
    assert "| strategy | urban_system_len | spatial_system_len |" in content
    assert "## 8. 注册表映射" in content
    assert "## 1.1 本次生效策略证明" in content
    assert "urban requested_shot" in content


def test_generate_single_mode_effective_proof(tmp_path: Path):
    output = tmp_path / "single.md"
    generated = generate_injection_audit_md(
        output_path=output,
        urban_shot="zero",
        spatial_shot="one",
        compare_all=False,
    )
    content = generated.read_text(encoding="utf-8")
    assert "## 6. 单策略生效检查" in content
    assert "urban requested_shot: zero" in content
    assert "spatial requested_shot: one" in content
