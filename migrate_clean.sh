#!/bin/bash
# Script de Migração Automática - Substituir Arquivos Originais
# Uso: chmod +x migrate_clean.sh && ./migrate_clean.sh

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    MIGRAÇÃO AUTOMÁTICA - REPOSITÓRIO LIMPO                  ║"
echo "║              Substituindo Originais com Novas Implementações                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir
print_step() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Verificar se estamos no diretório correto
if [ ! -f "main.py" ]; then
    print_error "main.py não encontrado. Execute este script na raiz do seu projeto!"
    exit 1
fi

# PASSO 1: BACKUP
print_step "PASSO 1: Criando Backup dos Arquivos Originais"

if [ -d "backup_antigos" ]; then
    print_warning "Diretório backup_antigos já existe"
    read -p "Deseja sobrescrever? (s/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        print_warning "Migração cancelada"
        exit 1
    fi
    rm -rf backup_antigos
fi

mkdir -p backup_antigos
print_success "Diretório backup_antigos criado"

# Fazer backup dos arquivos importantes
if [ -f "model_wrapper.py" ]; then
    cp model_wrapper.py backup_antigos/
    print_success "Backup: model_wrapper.py"
fi

if [ -f "requirements.txt" ]; then
    cp requirements.txt backup_antigos/
    print_success "Backup: requirements.txt"
fi

if [ -f "config.json" ]; then
    cp config.json backup_antigos/
    print_success "Backup: config.json"
fi

echo ""

# PASSO 2: REMOVER ANTIGOS
print_step "PASSO 2: Removendo Arquivos Originais"

if [ -f "model_wrapper.py" ]; then
    rm model_wrapper.py
    print_success "Removido: model_wrapper.py"
fi

if [ -f "requirements.txt" ]; then
    rm requirements.txt
    print_success "Removido: requirements.txt"
fi

if [ -f "config.json" ]; then
    rm config.json
    print_success "Removido: config.json"
fi

echo ""

# PASSO 3: COPIAR NOVOS
print_step "PASSO 3: Copiando Novos Arquivos"

if [ ! -f "model_wrapper_impl.py" ]; then
    print_error "model_wrapper_impl.py não encontrado!"
    exit 1
fi

cp model_wrapper_impl.py model_wrapper.py
print_success "Criado: model_wrapper.py (a partir de model_wrapper_impl.py)"

if [ ! -f "requirements_updated.txt" ]; then
    print_error "requirements_updated.txt não encontrado!"
    exit 1
fi

cp requirements_updated.txt requirements.txt
print_success "Criado: requirements.txt (a partir de requirements_updated.txt)"

echo ""

# PASSO 4: GERAR CONFIGURAÇÃO
print_step "PASSO 4: Gerando Nova Configuração"

if [ ! -f "config_updated.py" ]; then
    print_error "config_updated.py não encontrado!"
    exit 1
fi

python3 config_updated.py

if [ -f "config_full.json" ]; then
    cp config_full.json config.json
    print_success "Criado: config.json (a partir de config_full.json)"
else
    print_error "Falha ao gerar config.json"
    exit 1
fi

echo ""

# PASSO 5: INSTALAR DEPENDÊNCIAS
print_step "PASSO 5: Instalando Dependências Atualizadas"

read -p "Deseja instalar/atualizar dependências agora? (s/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Ss]$ ]]; then
    pip install --upgrade -r requirements.txt
    print_success "Dependências instaladas"
else
    print_warning "Pule para instalar depois com: pip install -r requirements.txt"
fi

echo ""

# PASSO 6: VERIFICAR COMPATIBILIDADE
print_step "PASSO 6: Verificando Compatibilidade"

python3 -c "from model_wrapper import get_model_wrapper; print('✓ Import OK')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Import de model_wrapper funcionando"
else
    print_error "Problema ao importar model_wrapper"
    exit 1
fi

echo ""

# PASSO 7: SUGERIR LIMPEZA
print_step "PASSO 7: Limpeza (Opcional)"

echo -e "${YELLOW}Arquivos temporários que podem ser removidos:${NC}"
echo ""

files_to_remove=(
    "model_wrapper_impl.py"
    "requirements_updated.txt"
    "config_updated.py"
    "config_full.json"
    "config_minimal.json"
)

echo "Arquivos que podem ser removidos de forma segura:"
for file in "${files_to_remove[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
echo "Arquivos que você pode manter para referência:"
echo "  - examples_usage.py (exemplos de uso)"
echo "  - IMPLEMENTATION_GUIDE.md (documentação)"
echo "  - README_IMPLEMENTACOES.py (resumo)"
echo "  - SOLUCAO_COMPLETA.md (explicação)"
echo "  - MIGRATION_PLAN.txt (este plano)"
echo ""

read -p "Deseja remover os arquivos temporários agora? (s/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Ss]$ ]]; then
    for file in "${files_to_remove[@]}"; do
        if [ -f "$file" ]; then
            rm "$file"
            print_success "Removido: $file"
        fi
    done
    echo ""
else
    print_warning "Você pode remover manualmente depois"
fi

echo ""

# RESUMO FINAL
print_step "✓ MIGRAÇÃO CONCLUÍDA COM SUCESSO!"

echo -e "${GREEN}Resumo:${NC}"
echo "  ✓ Backup criado em: backup_antigos/"
echo "  ✓ model_wrapper.py atualizado (com todas as features)"
echo "  ✓ requirements.txt atualizado (versões otimizadas)"
echo "  ✓ config.json criado (para pt-BR)"
echo ""

echo -e "${BLUE}Próximos passos:${NC}"
echo "  1. Rodar testes: python README_IMPLEMENTACOES.py test"
echo "  2. Testar frameworks: python examples_usage.py all"
echo "  3. Executar seu main.py normalmente"
echo ""

echo -e "${YELLOW}Seu código NÃO precisa de mudanças! 🎉${NC}"
echo "  from model_wrapper import get_model_wrapper  # Continua igual!"
echo ""

echo -e "${GREEN}Repositório está limpo e pronto para dissertação! 📚${NC}"
echo ""