import random


def pedra_papel_tesoura():
    print("Vamos jogar pedra-papel-tesoura\n")

    r = "pedra"
    p = "papel"
    t = "tesoura"
    all_choices = (r, p, t)

    jogador = input(f"Escolha uma opção ({r}, {p}, {t}): ")

    if jogador not in all_choices:
        print("Escolha inválida!\n")
        return

    computador = random.choice(all_choices)
    print(f"Você escolhe {jogador}, o computador escolheu {computador}.")

    if jogador == computador:
        print("Empate!\n")

    elif ((jogador == r and computador == t) or (jogador == p and computador == r) or (jogador == t and computador == p)):
        print("Você ganhou!\n")

    else:
        print("Você perdeu!\n")
