from datasets import load_dataset
import random

all_labels = [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            "atm_support",
            "automatic_top_up",
            "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit",
            "beneficiary_not_allowed",
            "cancel_transfer",
            "card_about_to_expire",
            "card_acceptance",
            "card_arrival",
            "card_delivery_estimate",
            "card_linking",
            "card_not_working",
            "card_payment_fee_charged",
            "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate",
            "card_swallowed",
            "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised",
            "change_pin",
            "compromised_card",
            "contactless_not_working",
            "country_support",
            "declined_card_payment",
            "declined_cash_withdrawal",
            "declined_transfer",
            "direct_debit_payment_not_recognised",
            "disposable_card_limits",
            "edit_personal_details",
            "exchange_charge",
            "exchange_rate",
            "exchange_via_app",
            "extra_charge_on_statement",
            "failed_transfer",
            "fiat_currency_support",
            "get_disposable_virtual_card",
            "get_physical_card",
            "getting_spare_card",
            "getting_virtual_card",
            "lost_or_stolen_card",
            "lost_or_stolen_phone",
            "order_physical_card",
            "passcode_forgotten",
            "pending_card_payment",
            "pending_cash_withdrawal",
            "pending_top_up",
            "pending_transfer",
            "pin_blocked",
            "receiving_money",
            "Refund_not_showing_up",
            "request_refund",
            "reverted_card_payment?",
            "supported_cards_and_currencies",
            "terminate_account",
            "top_up_by_bank_transfer_charge",
            "top_up_by_card_charge",
            "top_up_by_cash_or_cheque",
            "top_up_failed",
            "top_up_limits",
            "top_up_reverted",
            "topping_up_by_card",
            "transaction_charged_twice",
            "transfer_fee_charged",
            "transfer_into_account",
            "transfer_not_received_by_recipient",
            "transfer_timing",
            "unable_to_verify_identity",
            "verify_my_identity",
            "verify_source_of_funds",
            "verify_top_up",
            "virtual_card_not_working",
            "visa_or_mastercard",
            "why_verify_identity",
            "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal"
        ]

def select_banking77_data(given_dataset, total_count, seed=0):
    random.seed(seed)
    label_to_data_dict = {}
    for data in given_dataset:
        if data['label'] in label_to_data_dict:
            label_to_data_dict[data['label']].append(data)
        else:
            label_to_data_dict[data['label']] = [data]

    for key in label_to_data_dict:
        random.shuffle(label_to_data_dict[key])

    selected_data_list = []
    data_label_list = list(label_to_data_dict.keys())
    selected_label_to_count = {key: 0 for key in data_label_list}

    while len(selected_data_list) < total_count:
        for key in data_label_list:
            if len(selected_data_list) >= total_count:
                break
            if selected_label_to_count[key] < len(label_to_data_dict[key]):
                selected_data_list.append(label_to_data_dict[key][selected_label_to_count[key]])
                selected_label_to_count[key] += 1
        if all(selected_label_to_count[key] >= len(label_to_data_dict[key]) for key in data_label_list):
            break

    return selected_data_list

def format_banking77_embed(data):
    return "service query: " + data['text'] + "\nintent category: " + all_labels[data['label']]

def format_zerobanking77_prompt(test):

    prompt = "I am going to provide a customer service query and I want you to predict the intent of the query. Give only the intent of the query, and no extra commentary, formatting, or chattiness."
    prompt = prompt + 'You can only make prediction from the following categories: '
    for i, word in enumerate(all_labels):
        if i != len(all_labels) - 1:
            prompt = prompt + word + ', '
        else:
            prompt = prompt + word + '.\n'
    prompt = prompt + "service query: " + test['text'] + "\nintent category: "

    return prompt


def format_banking77_prompt(labeled, unlabled, test):

    prompt = 'Given a customer service query, please predict the intent of the query. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "service query: " + data['text'] + "\nintent category: " + all_labels[
            data['label']] + '\n'
    if len(unlabled) > 0:
        prompt = "Here are several examples of service query without the ground truth intent of the query. \n"
        for data in unlabled:
            prompt = prompt + "service query: " + data['text'] + '\n'

    #prompt = prompt + 'Given a customer service query, please predict the intent of the query. The predict answer must come from the demonstration examples with the exact format.'
    prompt = prompt + "I am going to provide another customer service query and I want you to predict the intent of the query. Give only the intent of the query, and no extra commentary, formatting, or chattiness."
    prompt = prompt + 'You can only make prediction from the following categories: '
    for i, word in enumerate(all_labels):
        if i != len(all_labels) - 1:
            prompt = prompt + word + ', '
        else:
            prompt = prompt + word + '.\n'
    prompt = prompt + "service query: " + test['text'] + "\nintent category: "

    return prompt

def format_pseudobanking77_prompt(labeled, pseudolabled, test):

    prompt = 'Given a customer service query, please predict the intent of the query. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "service query: " + data['text'] + "\nintent category: " + all_labels[data['label']] + '\n'
    if len(pseudolabled) > 0:
        for data in pseudolabled:
            prompt = prompt + "service query: " + data['example']['text'] + "\nintent category: " + data['pred'] + '\n'

    prompt = prompt + "I am going to provide another customer service query and I want you to predict the intent of the query. Give only the intent of the query, and no extra commentary, formatting, or chattiness."
    prompt = prompt + 'You can only make prediction from the following categories: '
    for i, word in enumerate(all_labels):
        if i != len(all_labels) - 1:
            prompt = prompt + word + ', '
        else:
            prompt = prompt + word + '.\n'
    prompt = prompt + "service query: " + test['text'] + "\nintent category: "

    return prompt


def calculate_banking77_acc(response, tem):

    temp_prompt = "intent category:"
    if tem['text'] not in response:
        response = response.split("service query")[0].strip()
    else:
        response = list(response.split(tem['text']))[-1].strip().split(temp_prompt)
        if len(response) > 1:
            response = response[1].split("service query")[0].strip()
        else:
            response = response[0].strip()
    response = response.strip().split("\n")[0]

    response = response.lower().strip()
    label = all_labels[tem['label']]
    label = label.lower()
    if response == label:
        return 1
    else:
        return 0


